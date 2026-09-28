import Foundation
import MLX
import MLXNN

// Qwen2.5-VL's vision tower, after mflux's `qwen_vision_*` and `qwen_patch_merger`: the picture in
// 14-pixel patches of two identical frames, 32 blocks with 2D rotary positions that attend within
// windows of 8 × 8 patches except in four full-attention blocks, then each 2 × 2 patches merged
// into one token of the language model's width. The reference feeds it float32 pixels, so it runs
// in float32 against its bf16 weights. Names follow mflux (`encoder.visual.*` in the checkpoint).

/// A picture's patch grid: frames, rows and columns of 14-pixel patches.
public struct QwenVisionGrid: Equatable, Sendable {
    public var t: Int
    public var h: Int
    public var w: Int

    public init(t: Int, h: Int, w: Int) {
        self.t = t
        self.h = h
        self.w = w
    }

    public var patches: Int { t * h * w }
}

/// A 3D convolution whose kernel is its stride: one row of flattened pixels per patch.
final class QwenVisionPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv3d
    let vision: QwenImageConfig.Vision

    init(_ vision: QwenImageConfig.Vision) {
        self.vision = vision
        let kernel = IntOrTriple((vision.temporalPatchSize, vision.patchSize, vision.patchSize))
        _proj.wrappedValue = Conv3d(
            inputChannels: vision.inChannels, outputChannels: vision.embedDim,
            kernelSize: kernel, stride: kernel, padding: 0, bias: false
        )
        super.init()
    }

    /// [N, C·T·P·P] rows, channels outermost → [N, embed].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let count = x.shape[0]
        let patches = x.reshaped([count, vision.inChannels, vision.temporalPatchSize, vision.patchSize, vision.patchSize])
            .transposed(0, 2, 3, 4, 1)
        return proj(patches).reshaped([count, vision.embedDim])
    }
}

final class QwenVisionAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    let heads: Int
    let headDim: Int

    init(embedDim: Int, heads: Int) {
        self.heads = heads
        headDim = embedDim / heads
        _qkv.wrappedValue = Linear(embedDim, 3 * embedDim, bias: true)
        _proj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        super.init()
    }

    /// `x` [S, embed]; `cos`/`sin` [S, headDim] in float32; `boundaries` where the runs of tokens
    /// that attend among themselves start and end (a single run: plain attention).
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, boundaries: [Int]) -> MLXArray {
        let (length, dim) = (x.shape[0], x.shape[1])
        let parts = qkv(x).reshaped([length, 3, heads, headDim])
        let q = Self.rope(parts[0..., 0].transposed(1, 0, 2), cos: cos, sin: sin)
        let k = Self.rope(parts[0..., 1].transposed(1, 0, 2), cos: cos, sin: sin)
        let v = parts[0..., 2].transposed(1, 0, 2)
        let scale = Float(1 / Double(headDim).squareRoot())
        let attended: MLXArray
        if boundaries.count > 2 {
            var outputs: [MLXArray] = []
            for index in 0 ..< boundaries.count - 1 {
                let run = boundaries[index] ..< boundaries[index + 1]
                let out = MLXFast.scaledDotProductAttention(
                    queries: q[0..., run].expandedDimensions(axis: 0),
                    keys: k[0..., run].expandedDimensions(axis: 0),
                    values: v[0..., run].expandedDimensions(axis: 0),
                    scale: scale, mask: nil
                )
                outputs.append(out.squeezed(axis: 0))
            }
            attended = concatenated(outputs, axis: 1)
        } else {
            attended = MLXFast.scaledDotProductAttention(
                queries: q.expandedDimensions(axis: 0), keys: k.expandedDimensions(axis: 0),
                values: v.expandedDimensions(axis: 0), scale: scale, mask: nil
            ).squeezed(axis: 0)
        }
        return proj(attended.transposed(1, 0, 2).reshaped([length, dim]))
    }

    /// Rotate-half rotary in float32 over [heads, S, headDim], back in the input's dtype.
    static func rope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let x32 = x.asType(.float32)
        let rotated = x32 * cos.expandedDimensions(axis: 0) + Qwen25Attention.rotateHalf(x32) * sin.expandedDimensions(axis: 0)
        return rotated.asType(x.dtype)
    }
}

final class QwenVisionMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dim: Int, hiddenDim: Int) {
        _gateProj.wrappedValue = Linear(dim, hiddenDim, bias: true)
        _upProj.wrappedValue = Linear(dim, hiddenDim, bias: true)
        _downProj.wrappedValue = Linear(hiddenDim, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class QwenVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "attn") var attn: QwenVisionAttention
    @ModuleInfo(key: "mlp") var mlp: QwenVisionMLP

    init(_ vision: QwenImageConfig.Vision) {
        _norm1.wrappedValue = RMSNorm(dimensions: vision.embedDim, eps: 1e-6)
        _norm2.wrappedValue = RMSNorm(dimensions: vision.embedDim, eps: 1e-6)
        _attn.wrappedValue = QwenVisionAttention(embedDim: vision.embedDim, heads: vision.numHeads)
        _mlp.wrappedValue = QwenVisionMLP(dim: vision.embedDim, hiddenDim: vision.mlpHiddenDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, boundaries: [Int]) -> MLXArray {
        let h = x + attn(norm1(x), cos: cos, sin: sin, boundaries: boundaries)
        return h + mlp(norm2(h))
    }
}

/// Four neighbouring patches (consecutive in the window order) into one token.
final class QwenVisionMerger: Module {
    @ModuleInfo(key: "ln_q") var lnQ: RMSNorm
    @ModuleInfo(key: "mlp_0") var mlp0: Linear
    @ModuleInfo(key: "mlp_1") var mlp1: Linear

    let mergedDim: Int

    init(contextDim: Int, outDim: Int, mergeSize: Int) {
        mergedDim = contextDim * mergeSize * mergeSize
        _lnQ.wrappedValue = RMSNorm(dimensions: contextDim, eps: 1e-6)
        _mlp0.wrappedValue = Linear(mergedDim, mergedDim, bias: true)
        _mlp1.wrappedValue = Linear(mergedDim, outDim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        mlp1(gelu(mlp0(lnQ(x).reshaped([-1, mergedDim]))))
    }
}

public final class Qwen25VisionTower: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: QwenVisionPatchEmbed
    @ModuleInfo(key: "blocks") var blocks: [QwenVisionBlock]
    @ModuleInfo(key: "merger") var merger: QwenVisionMerger

    public let vision: QwenImageConfig.Vision

    public init(_ vision: QwenImageConfig.Vision, outDim: Int) {
        self.vision = vision
        _patchEmbed.wrappedValue = QwenVisionPatchEmbed(vision)
        _blocks.wrappedValue = (0 ..< vision.depth).map { _ in QwenVisionBlock(vision) }
        _merger.wrappedValue = QwenVisionMerger(contextDim: vision.embedDim, outDim: outDim, mergeSize: vision.spatialMergeSize)
        super.init()
    }

    /// `pixelValues` [N, C·T·P·P], the patches of the pictures `grids` describes one after the
    /// other → [N / 4, outDim], one row per merged token in the same order.
    public func callAsFunction(_ pixelValues: MLXArray, grids: [QwenVisionGrid]) -> MLXArray {
        var hidden = patchEmbed(pixelValues.asType(.float32))
        let count = hidden.shape[0]
        let unit = vision.spatialMergeSize * vision.spatialMergeSize
        let groups = count / unit

        // Tokens regrouped window by window (four patches of a merged token stay together), their
        // rotary angles with them.
        let (windowIndex, windowBounds) = Self.windows(grids, vision: vision)
        let order = MLXArray(windowIndex.map { Int32($0) })
        hidden = hidden.reshaped([groups, unit, -1])[order].reshaped([count, -1])
        let angles = rotaryAngles(grids).reshaped([groups, unit, -1])[order].reshaped([count, -1])
        let doubled = concatenated([angles, angles], axis: -1)
        let (cosines, sines) = (cos(doubled), sin(doubled))

        var fullBounds = [0]
        for grid in grids { fullBounds.append(fullBounds[fullBounds.count - 1] + grid.patches) }
        for (index, block) in blocks.enumerated() {
            let bounds = vision.fullAttentionBlocks.contains(index) ? fullBounds : windowBounds
            hidden = block(hidden, cos: cosines, sin: sines, boundaries: bounds)
        }

        // Merged tokens back in reading order (`argsort` of the window order).
        var reverse = [Int32](repeating: 0, count: windowIndex.count)
        for (position, original) in windowIndex.enumerated() { reverse[original] = Int32(position) }
        return merger(hidden)[MLXArray(reverse)]
    }

    /// `rot_pos_emb`: each patch's row and column angles, [N, headDim / 2] in float32, patches in
    /// the order of the pixel rows (merge units of 2 × 2 together).
    func rotaryAngles(_ grids: [QwenVisionGrid]) -> MLXArray {
        let merge = vision.spatialMergeSize
        var rows: [Int32] = []
        var columns: [Int32] = []
        for grid in grids {
            var frameRows: [Int32] = []
            var frameColumns: [Int32] = []
            for blockRow in 0 ..< grid.h / merge {
                for blockColumn in 0 ..< grid.w / merge {
                    for i in 0 ..< merge {
                        for j in 0 ..< merge {
                            frameRows.append(Int32(blockRow * merge + i))
                            frameColumns.append(Int32(blockColumn * merge + j))
                        }
                    }
                }
            }
            for _ in 0 ..< grid.t {
                rows += frameRows
                columns += frameColumns
            }
        }
        let largest = grids.map { max($0.h, $0.w) }.max() ?? 1
        // `VisionRotaryEmbedding(head_dim / 2)`: inverse frequencies over half the rotary width.
        let dim = vision.headDim / 2
        let exponents = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) }) / Float(dim)
        let inverse = 1 / pow(Float(10000), exponents)
        let table = outer(MLXArray((0 ..< largest).map { Float($0) }), inverse)
        let rowAngles = table[MLXArray(rows)]
        let columnAngles = table[MLXArray(columns)]
        return stacked([rowAngles, columnAngles], axis: 1).reshaped([rows.count, -1])
    }

    /// `get_window_index`: the merged tokens window by window (windows of `window_size / patch /
    /// merge` merged tokens a side, each picture padded to whole windows), and the boundaries of
    /// the windows in patches, empty windows dropped.
    static func windows(_ grids: [QwenVisionGrid], vision: QwenImageConfig.Vision) -> (index: [Int], bounds: [Int]) {
        let merge = vision.spatialMergeSize
        let side = vision.windowSize / vision.patchSize / merge
        var index: [Int] = []
        var bounds = [0]
        var offset = 0
        for grid in grids {
            let (rows, columns) = (grid.h / merge, grid.w / merge)
            let (windowRows, windowColumns) = ((rows + side - rows % side) / side, (columns + side - columns % side) / side)
            for t in 0 ..< grid.t {
                for windowRow in 0 ..< windowRows {
                    for windowColumn in 0 ..< windowColumns {
                        var tokens = 0
                        for y in 0 ..< side {
                            for x in 0 ..< side {
                                let (row, column) = (windowRow * side + y, windowColumn * side + x)
                                guard row < rows, column < columns else { continue }
                                index.append(offset + t * rows * columns + row * columns + column)
                                tokens += 1
                            }
                        }
                        bounds.append(bounds[bounds.count - 1] + tokens * merge * merge)
                    }
                }
            }
            offset += grid.t * rows * columns
        }
        var unique = [bounds[0]]
        for bound in bounds.dropFirst() where bound != unique[unique.count - 1] { unique.append(bound) }
        return (index, unique)
    }
}
