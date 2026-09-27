import Foundation
import MLX
import MLXNN

// Mixture-of-experts layers after mlx-lm's `switch_layers.py` (which mflux's Ling MoE encoder
// uses, via `gpt_oss/switch_layers.py`) and mlx-swift-lm's port: every expert's weights stacked
// in one array, applied with gathered matrix multiplications.

/// A stack of `numExperts` linear layers, `weight` [E, out, in], applied to the experts an index
/// array picks for each row.
open class SwitchLinear: Module, Quantizable {
    public let weight: MLXArray
    public let bias: MLXArray?
    public let inputDims: Int
    public let outputDims: Int
    public let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = false) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts
        // Replaced by the checkpoint before anything is evaluated.
        weight = MLXArray.zeros([numExperts, outputDims, inputDims], dtype: .bfloat16)
        self.bias = bias ? MLXArray.zeros([numExperts, outputDims], dtype: .bfloat16) : nil
        super.init()
    }

    /// For subclasses that bring their own arrays.
    public init(inputDims: Int, outputDims: Int, numExperts: Int, weight: MLXArray, bias: MLXArray?) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts
        self.weight = weight
        self.bias = bias
        super.init()
    }

    /// `x` [..., 1, in] with `indices` [...] of experts → [..., 1, out].
    open func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false) -> MLXArray {
        var result = gatherMM(x, weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sortedIndices)
        if let bias {
            result = result + bias[indices].expandedDimensions(axis: -2)
        }
        return result
    }

    public func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

/// `SwitchLinear` with quantized experts: `weight` packed, `scales` and `biases` per group.
open class QuantizedSwitchLinear: SwitchLinear, Quantized {
    public let scales: MLXArray
    public let biases: MLXArray?
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(_ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        let (quantizedWeight, scales, biases) = MLX.quantized(other.weight, groupSize: groupSize, bits: bits, mode: mode)
        self.scales = scales
        self.biases = biases
        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias
        )
        freeze()
    }

    open override func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false) -> MLXArray {
        var result = gatherQuantizedMM(
            x, weight, scales: scales, biases: biases, rhsIndices: indices, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode, sortedIndices: sortedIndices
        )
        if let bias {
            result = result + bias[indices].expandedDimensions(axis: -2)
        }
        return result
    }
}

/// `SwitchGLU`: gated experts, `down(silu(gate(x)) · up(x))` for each expert a row is routed to.
public final class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    public init(inputDims: Int, hiddenDims: Int, numExperts: Int) {
        _gateProj.wrappedValue = SwitchLinear(inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts)
        _upProj.wrappedValue = SwitchLinear(inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts)
        _downProj.wrappedValue = SwitchLinear(inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts)
        super.init()
    }

    /// `x` [N, in], `indices` [N, K] → [N, K, in]: each row through its K experts, unweighted.
    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = x.expandedDimensions(axes: [-2, -3])
        // With many rows, sort them by expert so each expert's weights are read once.
        let doSort = indices.size >= 64
        var idx = indices
        var inverseOrder: MLXArray?
        if doSort {
            (x, idx, inverseOrder) = Self.gatherSort(x, indices: indices)
        }
        let up = upProj(x, idx, sortedIndices: doSort)
        let gate = gateProj(x, idx, sortedIndices: doSort)
        x = downProj(silu(gate) * up, idx, sortedIndices: doSort)
        if let inverseOrder {
            x = x[inverseOrder]
            x = unflatten(x, axis: 0, shape: indices.shape)
        }
        return x.squeezed(axis: -2)
    }

    static func gatherSort(_ x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let k = indices.shape[indices.ndim - 1]
        let flat = indices.flattened()
        let order = argSort(flat)
        let inverseOrder = argSort(order)
        return (x.flattened(start: 0, end: -3)[order.floorDivide(k)], flat[order], inverseOrder)
    }
}
