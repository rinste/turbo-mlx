import Foundation
import MLX
import MLXNN

// BiRefNet (ZhengPeng7/BiRefNet, MIT) as idol-sdk packages it (`models/birefnet.py`): a Swin-L backbone run on the
// picture and on its half-size copy, a decoder of deformable ASPP blocks with gradient-gated attention, read at 1024 ×
// 1024; it gives the alpha Idol cuts the object out with. Channels-last throughout, float32 as the reference runs it
// away from CUDA; the tensors as released (convolutions turned channels-last on loading).

// MARK: Shared

/// `F.interpolate(mode="bilinear", align_corners=True)` on [B, H, W, C].
public func idolResizeBilinear(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
    func along(_ x: MLXArray, axis: Int, size: Int) -> MLXArray {
        let input = x.dim(axis)
        guard input != size else { return x }
        let scale: Float = size > 1 ? Float(input - 1) / Float(size - 1) : 0
        var low = [Int32](), high = [Int32](), weight = [Float]()
        for i in 0 ..< size {
            let source = scale * Float(i)
            let l = Swift.min(Int(source), input - 1)
            low.append(Int32(l))
            high.append(Int32(Swift.min(l + 1, input - 1)))
            weight.append(source - Float(l))
        }
        var shape = [Int](repeating: 1, count: x.ndim)
        shape[axis] = size
        let w = MLXArray(weight).reshaped(shape)
        let a = take(x, MLXArray(low), axis: axis)
        let b = take(x, MLXArray(high), axis: axis)
        return a * (1 - w) + b * w
    }
    return along(along(x, axis: 1, size: height), axis: 2, size: width)
}

/// A BatchNorm2d in inference: the running statistics, eps 1e-5.
final class IdolBatchNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray

    init(_ channels: Int) {
        _weight.wrappedValue = MLXArray.ones([channels])
        _bias.wrappedValue = MLXArray.zeros([channels])
        _runningMean.wrappedValue = MLXArray.zeros([channels])
        _runningVar.wrappedValue = MLXArray.ones([channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x - runningMean) / sqrt(runningVar + 1e-5) * weight + bias
    }
}

/// A `Sequential(Conv2d, BatchNorm2d, ReLU)` (or the head's lone convolution): the checkpoint's `0` and `1`.
final class IdolConvBN: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "bn") var bn: IdolBatchNorm?

    init(_ input: Int, _ output: Int, kernel: Int, bias: Bool, norm: Bool) {
        _conv.wrappedValue = Conv2d(inputChannels: input, outputChannels: output, kernelSize: .init(kernel), padding: .init(kernel / 2), bias: bias)
        _bn.wrappedValue = norm ? IdolBatchNorm(output) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let bn else { return conv(x) }
        return relu(bn(conv(x)))
    }
}

// MARK: Swin-L

final class IdolSwinMlp: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ dim: Int) {
        _fc1.wrappedValue = Linear(dim, 4 * dim)
        _fc2.wrappedValue = Linear(4 * dim, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

final class IdolWindowAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "relative_position_bias_table") var biasTable: MLXArray
    let heads: Int
    let window: Int

    init(dim: Int, heads: Int, window: Int) {
        self.heads = heads
        self.window = window
        _qkv.wrappedValue = Linear(dim, 3 * dim)
        _proj.wrappedValue = Linear(dim, dim)
        _biasTable.wrappedValue = MLXArray.zeros([(2 * window - 1) * (2 * window - 1), heads])
        super.init()
    }

    /// The relative position bias [heads, N, N] of a window's N = w² tokens.
    func positionBias() -> MLXArray {
        let w = window
        var index = [Int32]()
        index.reserveCapacity(w * w * w * w)
        for a in 0 ..< w * w {
            for b in 0 ..< w * w {
                let dy = a / w - b / w + w - 1
                let dx = a % w - b % w + w - 1
                index.append(Int32(dy * (2 * w - 1) + dx))
            }
        }
        return take(biasTable, MLXArray(index), axis: 0).reshaped([w * w, w * w, heads]).transposed(2, 0, 1)
    }

    /// x [windows, N, C]; `mask` [windows per picture, N, N] for shifted windows.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (count, n, c) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = c / heads
        let parts = qkv(x).reshaped([count, n, 3, heads, headDim]).transposed(2, 0, 3, 1, 4)
        let (q, k, v) = (parts[0] * (1 / Float(headDim).squareRoot()), parts[1], parts[2])
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) + positionBias().expandedDimensions(axis: 0)
        if let mask {
            let windows = mask.dim(0)
            scores = (scores.reshaped([count / windows, windows, heads, n, n]) + mask.expandedDimensions(axes: [0, 2]))
                .reshaped([count, heads, n, n])
        }
        let attended = matmul(softmax(scores, axis: -1), v)
        return proj(attended.transposed(0, 2, 1, 3).reshaped([count, n, c]))
    }
}

final class IdolSwinBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: IdolWindowAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: IdolSwinMlp
    let window: Int
    let shift: Int

    init(dim: Int, heads: Int, window: Int, shift: Int) {
        self.window = window
        self.shift = shift
        _norm1.wrappedValue = LayerNorm(dimensions: dim)
        _attn.wrappedValue = IdolWindowAttention(dim: dim, heads: heads, window: window)
        _norm2.wrappedValue = LayerNorm(dimensions: dim)
        _mlp.wrappedValue = IdolSwinMlp(dim)
        super.init()
    }

    /// x [B, H, W, C].
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let (batch, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let w = window
        let paddedHeight = (height + w - 1) / w * w, paddedWidth = (width + w - 1) / w * w
        var h = norm1(x)
        if paddedHeight != height || paddedWidth != width {
            h = padded(h, widths: [.init(0), .init((0, paddedHeight - height)), .init((0, paddedWidth - width)), .init(0)])
        }
        if shift > 0 { h = Self.roll(h, by: -shift) }
        let windows = h.reshaped([batch, paddedHeight / w, w, paddedWidth / w, w, channels]).transposed(0, 1, 3, 2, 4, 5)
            .reshaped([-1, w * w, channels])
        let attended = attn(windows, mask: shift > 0 ? mask : nil)
        h = attended.reshaped([batch, paddedHeight / w, paddedWidth / w, w, w, channels]).transposed(0, 1, 3, 2, 4, 5)
            .reshaped([batch, paddedHeight, paddedWidth, channels])
        if shift > 0 { h = Self.roll(h, by: shift) }
        if paddedHeight != height || paddedWidth != width { h = h[0..., 0 ..< height, 0 ..< width, 0...] }
        let x = x + h
        return x + mlp(norm2(x))
    }

    /// `torch.roll` over rows and columns: element i takes element (i − shift) mod n.
    static func roll(_ x: MLXArray, by shift: Int) -> MLXArray {
        func along(_ x: MLXArray, axis: Int) -> MLXArray {
            let n = x.dim(axis)
            let s = ((shift % n) + n) % n
            guard s != 0 else { return x }
            let head = axis == 1 ? x[0..., (n - s)...] : x[0..., 0..., (n - s)...]
            let tail = axis == 1 ? x[0..., 0 ..< (n - s)] : x[0..., 0..., 0 ..< (n - s)]
            return concatenated([head, tail], axis: axis)
        }
        return along(along(x, axis: 1), axis: 2)
    }
}

final class IdolPatchMerging: Module {
    @ModuleInfo(key: "reduction") var reduction: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ dim: Int) {
        _reduction.wrappedValue = Linear(4 * dim, 2 * dim, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: 4 * dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x[:, dy::2, dx::2] for (dy, dx) = (0, 0), (1, 0), (0, 1), (1, 1).
        let (batch, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let grid = x.reshaped([batch, height / 2, 2, width / 2, 2, channels])
        let parts = [(0, 0), (1, 0), (0, 1), (1, 1)].map { grid[0..., 0..., $0.0, 0..., $0.1] }
        return reduction(norm(concatenated(parts, axis: -1)))
    }
}

final class IdolSwinLayer: Module {
    @ModuleInfo(key: "blocks") var blocks: [IdolSwinBlock]
    @ModuleInfo(key: "downsample") var downsample: IdolPatchMerging?
    let window: Int

    init(dim: Int, depth: Int, heads: Int, window: Int, downsample: Bool) {
        self.window = window
        _blocks.wrappedValue = (0 ..< depth).map { IdolSwinBlock(dim: dim, heads: heads, window: window, shift: $0 % 2 == 0 ? 0 : window / 2) }
        _downsample.wrappedValue = downsample ? IdolPatchMerging(dim) : nil
        super.init()
    }

    /// The shifted windows' mask [windows, N, N] for a padded grid: −100 between tokens of different regions.
    func shiftMask(height: Int, width: Int) -> MLXArray {
        let w = window, s = window / 2
        let paddedHeight = (height + w - 1) / w * w, paddedWidth = (width + w - 1) / w * w
        func region(_ i: Int, _ n: Int) -> Int { i < n - w ? 0 : i < n - s ? 1 : 2 }
        var ids = [Int](repeating: 0, count: paddedHeight * paddedWidth)
        for y in 0 ..< paddedHeight {
            for x in 0 ..< paddedWidth { ids[y * paddedWidth + x] = 3 * region(y, paddedHeight) + region(x, paddedWidth) }
        }
        let (rows, columns) = (paddedHeight / w, paddedWidth / w)
        var mask = [Float](repeating: 0, count: rows * columns * w * w * w * w)
        for wy in 0 ..< rows {
            for wx in 0 ..< columns {
                let window = wy * columns + wx
                let tokens = (0 ..< w * w).map { ids[(wy * w + $0 / w) * paddedWidth + wx * w + $0 % w] }
                for a in 0 ..< w * w {
                    for b in 0 ..< w * w where tokens[a] != tokens[b] {
                        mask[(window * w * w + a) * w * w + b] = -100
                    }
                }
            }
        }
        return MLXArray(mask).reshaped([rows * columns, w * w, w * w])
    }

    /// Returns the layer's output (before merging) and the merged input of the next layer.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let mask = shiftMask(height: x.dim(1), width: x.dim(2))
        var h = x
        for block in blocks { h = block(h, mask: mask) }
        return (h, downsample.map { $0(h) } ?? h)
    }
}

final class IdolSwinPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm

    override init() {
        _proj.wrappedValue = Conv2d(inputChannels: 3, outputChannels: 192, kernelSize: 4, stride: 4)
        _norm.wrappedValue = LayerNorm(dimensions: 192)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(proj(x)) }
}

/// `_SwinLarge`: embed 192, depths 2, 2, 18, 2, heads 6, 12, 24, 48, windows of 12.
final class IdolSwinLarge: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: IdolSwinPatchEmbed
    @ModuleInfo(key: "layers") var layers: [IdolSwinLayer]
    @ModuleInfo(key: "norm0") var norm0: LayerNorm
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm

    override init() {
        let depths = [2, 2, 18, 2], heads = [6, 12, 24, 48]
        _patchEmbed.wrappedValue = IdolSwinPatchEmbed()
        _layers.wrappedValue = (0 ..< 4).map { IdolSwinLayer(dim: 192 << $0, depth: depths[$0], heads: heads[$0], window: 12, downsample: $0 < 3) }
        _norm0.wrappedValue = LayerNorm(dimensions: 192)
        _norm1.wrappedValue = LayerNorm(dimensions: 384)
        _norm2.wrappedValue = LayerNorm(dimensions: 768)
        _norm3.wrappedValue = LayerNorm(dimensions: 1536)
        super.init()
    }

    /// x [B, H, W, 3] → the four stages' features, normalized.
    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        var h = patchEmbed(x)
        var outs = [MLXArray]()
        for (layer, norm) in zip(layers, [norm0, norm1, norm2, norm3]) {
            let (out, next) = layer(h)
            outs.append(norm(out))
            h = next
        }
        return outs
    }
}

// MARK: Decoder

/// torchvision's `deform_conv2d` with a modulator (DCNv2) for stride 1, padding k / 2 and no dilation: each output
/// position samples the input at its taps moved by predicted offsets (bilinear, zero outside), weighted by the taps'
/// modulators, then one matrix product. The columns are written by one Metal pass (as mlx-serve's `deform_cols`),
/// a block of positions at a time.
enum IdolDeformable {
    static let kernel = MLXFast.metalKernel(
        name: "idol_deform_columns",
        inputNames: ["X", "OFF", "MSK", "start"],
        outputNames: ["O"],
        source: """
            const uint c4 = thread_position_in_grid.x * 4;
            const uint p = thread_position_in_grid.y;
            const uint q = thread_position_in_grid.z;
            const uint C = X_shape[2];
            if (c4 >= C) return;
            const uint pos = q + uint(start);
            const int y = int(pos) / W, x = int(pos) % W;
            const int i = int(p) / KS, j = int(p) % KS;
            const float py = float(y + i - KS / 2) + OFF[size_t(pos) * 2 * KS * KS + 2 * p];
            const float px = float(x + j - KS / 2) + OFF[size_t(pos) * 2 * KS * KS + 2 * p + 1];
            const float fy = metal::floor(py), fx = metal::floor(px);
            const float ly = py - fy, lx = px - fx;
            float4 acc = 0;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    const int yy = int(fy) + dy, xx = int(fx) + dx;
                    if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
                    const float weight = (dy == 0 ? 1.0f - ly : ly) * (dx == 0 ? 1.0f - lx : lx);
                    acc += weight * float4(*(const device float4*)(X + size_t(yy * W + xx) * C + c4));
                }
            }
            const float m = MSK[size_t(pos) * KS * KS + p];
            *(device float4*)(O + size_t(q) * KS * KS * C + p * C + c4) = acc * m;
            """
    )

    /// x [1, H, W, C] (C a multiple of 4), offsets [1, H, W, 2k²] as (dy, dx) pairs per tap, modulators [1, H, W, k²],
    /// weight [Cout, k, k, C] → [1, H, W, Cout]; float32.
    static func apply(_ x: MLXArray, offsets: MLXArray, modulators: MLXArray, weight: MLXArray, kernelSize k: Int) -> MLXArray {
        let (height, width, channels) = (x.dim(1), x.dim(2), x.dim(3))
        let positions = height * width
        let taps = k * k
        let matrix = weight.reshaped([weight.dim(0), taps * channels]).transposed()
        // Columns of at most 256 MB at a time.
        let block = Swift.max(1, Swift.min(positions, (256 << 20) / (taps * channels * 4)))
        let input = x.asType(.float32).reshaped([height, width, channels])
        let off = offsets.asType(.float32).reshaped([-1])
        let mod = modulators.asType(.float32).reshaped([-1])
        var outputs = [MLXArray]()
        var start = 0
        while start < positions {
            let count = Swift.min(block, positions - start)
            let columns = kernel(
                [input, off, mod, MLXArray(Int32(start))],
                template: [("W", width), ("H", height), ("KS", k)],
                grid: (channels / 4, taps, count),
                threadGroup: (Swift.min(channels / 4, 32), 1, Swift.max(1, 256 / Swift.min(channels / 4, 32))),
                outputShapes: [[count, taps * channels]],
                outputDTypes: [.float32]
            )[0]
            outputs.append(matmul(columns, matrix))
            start += count
        }
        return concatenated(outputs, axis: 0).reshaped([1, height, width, weight.dim(0)])
    }
}

/// `_DeformableConv2d`: its offsets and modulators predicted by two plain convolutions of the same size.
final class IdolDeformableConv: Module {
    @ModuleInfo(key: "offset_conv") var offsetConv: Conv2d
    @ModuleInfo(key: "modulator_conv") var modulatorConv: Conv2d
    @ModuleInfo(key: "regular_conv") var regularConv: Conv2d
    let kernelSize: Int

    init(_ input: Int, _ output: Int, kernel: Int) {
        kernelSize = kernel
        _offsetConv.wrappedValue = Conv2d(inputChannels: input, outputChannels: 2 * kernel * kernel, kernelSize: .init(kernel), padding: .init(kernel / 2))
        _modulatorConv.wrappedValue = Conv2d(inputChannels: input, outputChannels: kernel * kernel, kernelSize: .init(kernel), padding: .init(kernel / 2))
        _regularConv.wrappedValue = Conv2d(inputChannels: input, outputChannels: output, kernelSize: .init(kernel), padding: .init(kernel / 2), bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let offsets = offsetConv(x)
        let modulators = 2 * sigmoid(modulatorConv(x))
        return IdolDeformable.apply(x, offsets: offsets, modulators: modulators, weight: regularConv.weight, kernelSize: kernelSize)
    }
}

final class IdolASPPModule: Module {
    @ModuleInfo(key: "atrous_conv") var atrousConv: IdolDeformableConv
    @ModuleInfo(key: "bn") var bn: IdolBatchNorm

    init(_ input: Int, _ output: Int, kernel: Int) {
        _atrousConv.wrappedValue = IdolDeformableConv(input, output, kernel: kernel)
        _bn.wrappedValue = IdolBatchNorm(output)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { relu(bn(atrousConv(x))) }
}

/// `_ASPPDeformable`: a 1 × 1 and three deformable branches (1, 3, 7) and the global average, side by side.
final class IdolASPP: Module {
    @ModuleInfo(key: "aspp1") var aspp1: IdolASPPModule
    @ModuleInfo(key: "aspp_deforms") var asppDeforms: [IdolASPPModule]
    @ModuleInfo(key: "global_avg_pool") var globalAvgPool: IdolConvBN
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: IdolBatchNorm

    init(_ channels: Int) {
        let inter = 256
        _aspp1.wrappedValue = IdolASPPModule(channels, inter, kernel: 1)
        _asppDeforms.wrappedValue = [1, 3, 7].map { IdolASPPModule(channels, inter, kernel: $0) }
        _globalAvgPool.wrappedValue = IdolConvBN(channels, inter, kernel: 1, bias: false, norm: true)
        _conv1.wrappedValue = Conv2d(inputChannels: inter * 5, outputChannels: channels, kernelSize: 1, bias: false)
        _bn1.wrappedValue = IdolBatchNorm(channels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x1 = aspp1(x)
        let deforms = asppDeforms.map { $0(x) }
        let pooled = globalAvgPool(x.mean(axes: [1, 2], keepDims: true))
        let x5 = idolResizeBilinear(pooled, height: x1.dim(1), width: x1.dim(2))
        return relu(bn1(conv1(concatenated([x1] + deforms + [x5], axis: -1))))
    }
}

final class IdolDecoderBlock: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "bn_in") var bnIn: IdolBatchNorm
    @ModuleInfo(key: "dec_att") var decAtt: IdolASPP
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    @ModuleInfo(key: "bn_out") var bnOut: IdolBatchNorm

    init(_ input: Int, _ output: Int) {
        _convIn.wrappedValue = Conv2d(inputChannels: input, outputChannels: 64, kernelSize: 3, padding: 1)
        _bnIn.wrappedValue = IdolBatchNorm(64)
        _decAtt.wrappedValue = IdolASPP(64)
        _convOut.wrappedValue = Conv2d(inputChannels: 64, outputChannels: output, kernelSize: 3, padding: 1)
        _bnOut.wrappedValue = IdolBatchNorm(output)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        bnOut(convOut(decAtt(relu(bnIn(convIn(x))))))
    }
}

final class IdolSimpleConvs: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(_ input: Int, _ output: Int) {
        _conv1.wrappedValue = Conv2d(inputChannels: input, outputChannels: 64, kernelSize: 3, padding: 1)
        _convOut.wrappedValue = Conv2d(inputChannels: 64, outputChannels: output, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { convOut(conv1(x)) }
}

final class IdolLateral: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(_ channels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

final class IdolMatteDecoder: Module {
    @ModuleInfo(key: "ipt_blk5") var iptBlk5: IdolSimpleConvs
    @ModuleInfo(key: "ipt_blk4") var iptBlk4: IdolSimpleConvs
    @ModuleInfo(key: "ipt_blk3") var iptBlk3: IdolSimpleConvs
    @ModuleInfo(key: "ipt_blk2") var iptBlk2: IdolSimpleConvs
    @ModuleInfo(key: "ipt_blk1") var iptBlk1: IdolSimpleConvs
    @ModuleInfo(key: "decoder_block4") var decoderBlock4: IdolDecoderBlock
    @ModuleInfo(key: "decoder_block3") var decoderBlock3: IdolDecoderBlock
    @ModuleInfo(key: "decoder_block2") var decoderBlock2: IdolDecoderBlock
    @ModuleInfo(key: "decoder_block1") var decoderBlock1: IdolDecoderBlock
    @ModuleInfo(key: "conv_out1") var convOut1: IdolConvBN
    @ModuleInfo(key: "lateral_block4") var lateralBlock4: IdolLateral
    @ModuleInfo(key: "lateral_block3") var lateralBlock3: IdolLateral
    @ModuleInfo(key: "lateral_block2") var lateralBlock2: IdolLateral
    @ModuleInfo(key: "gdt_convs_4") var gdtConvs4: IdolConvBN
    @ModuleInfo(key: "gdt_convs_3") var gdtConvs3: IdolConvBN
    @ModuleInfo(key: "gdt_convs_2") var gdtConvs2: IdolConvBN
    @ModuleInfo(key: "gdt_convs_attn_4") var gdtAttn4: IdolConvBN
    @ModuleInfo(key: "gdt_convs_attn_3") var gdtAttn3: IdolConvBN
    @ModuleInfo(key: "gdt_convs_attn_2") var gdtAttn2: IdolConvBN

    override init() {
        let c = [3072, 1536, 768, 384]
        _iptBlk5.wrappedValue = IdolSimpleConvs(1024 * 3, c[0] / 8)
        _iptBlk4.wrappedValue = IdolSimpleConvs(256 * 3, c[0] / 8)
        _iptBlk3.wrappedValue = IdolSimpleConvs(64 * 3, c[1] / 8)
        _iptBlk2.wrappedValue = IdolSimpleConvs(16 * 3, c[2] / 8)
        _iptBlk1.wrappedValue = IdolSimpleConvs(3, c[3] / 8)
        _decoderBlock4.wrappedValue = IdolDecoderBlock(c[0] + c[0] / 8, c[1])
        _decoderBlock3.wrappedValue = IdolDecoderBlock(c[1] + c[0] / 8, c[2])
        _decoderBlock2.wrappedValue = IdolDecoderBlock(c[2] + c[1] / 8, c[3])
        _decoderBlock1.wrappedValue = IdolDecoderBlock(c[3] + c[2] / 8, c[3] / 2)
        _convOut1.wrappedValue = IdolConvBN(c[3] / 2 + c[3] / 8, 1, kernel: 1, bias: true, norm: false)
        _lateralBlock4.wrappedValue = IdolLateral(c[1])
        _lateralBlock3.wrappedValue = IdolLateral(c[2])
        _lateralBlock2.wrappedValue = IdolLateral(c[3])
        _gdtConvs4.wrappedValue = IdolConvBN(c[1], 16, kernel: 3, bias: true, norm: true)
        _gdtConvs3.wrappedValue = IdolConvBN(c[2], 16, kernel: 3, bias: true, norm: true)
        _gdtConvs2.wrappedValue = IdolConvBN(c[3], 16, kernel: 3, bias: true, norm: true)
        _gdtAttn4.wrappedValue = IdolConvBN(16, 1, kernel: 1, bias: true, norm: false)
        _gdtAttn3.wrappedValue = IdolConvBN(16, 1, kernel: 1, bias: true, norm: false)
        _gdtAttn2.wrappedValue = IdolConvBN(16, 1, kernel: 1, bias: true, norm: false)
        super.init()
    }

    /// `_image2patches`: the picture [1, H, W, C] cut into the grid of `reference`'s size, the blocks stacked as
    /// channels (channel, block row, block column).
    static func patches(_ x: MLXArray, like reference: MLXArray) -> MLXArray {
        let (height, width, channels) = (x.dim(1), x.dim(2), x.dim(3))
        let (h, w) = (reference.dim(1), reference.dim(2))
        let (rows, columns) = (height / h, width / w)
        return x.reshaped([1, rows, h, columns, w, channels]).transposed(0, 2, 4, 5, 1, 3).reshaped([1, h, w, channels * rows * columns])
    }

    func callAsFunction(_ x: MLXArray, _ x1: MLXArray, _ x2: MLXArray, _ x3: MLXArray, _ x4: MLXArray) -> MLXArray {
        func resize(_ y: MLXArray, like z: MLXArray) -> MLXArray { idolResizeBilinear(y, height: z.dim(1), width: z.dim(2)) }
        let in4 = concatenated([x4, iptBlk5(Self.patches(x, like: x4))], axis: -1)
        var p4 = decoderBlock4(in4)
        p4 = p4 * sigmoid(gdtAttn4(gdtConvs4(p4)))
        var p3 = resize(p4, like: x3) + lateralBlock4(x3)
        p3 = concatenated([p3, iptBlk4(Self.patches(x, like: p3))], axis: -1)
        p3 = decoderBlock3(p3)
        p3 = p3 * sigmoid(gdtAttn3(gdtConvs3(p3)))
        var p2 = resize(p3, like: x2) + lateralBlock3(x2)
        p2 = concatenated([p2, iptBlk3(Self.patches(x, like: p2))], axis: -1)
        p2 = decoderBlock2(p2)
        p2 = p2 * sigmoid(gdtAttn2(gdtConvs2(p2)))
        var p1 = resize(p2, like: x1) + lateralBlock2(x1)
        p1 = concatenated([p1, iptBlk2(Self.patches(x, like: p1))], axis: -1)
        p1 = decoderBlock1(p1)
        p1 = resize(p1, like: x)
        p1 = concatenated([p1, iptBlk1(Self.patches(x, like: p1))], axis: -1)
        return convOut1(p1)
    }
}

/// `BiRefNet`: the picture's alpha.
public final class IdolMatte: Module {
    @ModuleInfo(key: "bb") var backbone: IdolSwinLarge
    @ModuleInfo(key: "squeeze_module") var squeezeModule: [IdolDecoderBlock]
    @ModuleInfo(key: "decoder") var decoder: IdolMatteDecoder
    static let size = 1024
    public static let mean: [Float] = [0.485, 0.456, 0.406]
    public static let std: [Float] = [0.229, 0.224, 0.225]

    override init() {
        _backbone.wrappedValue = IdolSwinLarge()
        _squeezeModule.wrappedValue = [IdolDecoderBlock(3072 + 384 + 768 + 1536, 3072)]
        _decoder.wrappedValue = IdolMatteDecoder()
        super.init()
    }

    /// `birefnet.safetensors` as released: convolutions turned channels-last, `Sequential` layers (`.0` the
    /// convolution, `.1` its batch norm) given their names, the stored position indices and batch counters left out.
    public static func load(url: URL) throws -> IdolMatte {
        let model = IdolMatte()
        var tensors: [String: MLXArray] = [:]
        for (key, value) in try loadArrays(url: url) {
            if key.hasSuffix("relative_position_index") || key.hasSuffix("num_batches_tracked") { continue }
            var value = value.asType(.float32)
            if value.ndim == 4 { value = value.transposed(0, 2, 3, 1) }
            tensors[renamed(key)] = value
        }
        try WeightLoading.apply(tensors, to: model)
        eval(model)
        return model
    }

    static func renamed(_ key: String) -> String {
        var key = key
        for name in ["global_avg_pool", "gdt_convs_4", "gdt_convs_3", "gdt_convs_2", "gdt_convs_attn_4", "gdt_convs_attn_3",
                     "gdt_convs_attn_2", "conv_out1"] {
            // `global_avg_pool` is (pool, conv, bn, relu): its conv is `.1`, its norm `.2`; the others (conv, bn, relu).
            let (conv, norm) = name == "global_avg_pool" ? (".1.", ".2.") : (".0.", ".1.")
            key = key.replacingOccurrences(of: "\(name)\(conv)", with: "\(name).conv.")
                .replacingOccurrences(of: "\(name)\(norm)", with: "\(name).bn.")
        }
        return key
    }

    /// The backbone's four stages for a normalized input (`verify`).
    public func backboneFeatures(_ x: MLXArray) -> [MLXArray] { backbone(x) }

    /// The network's alpha map [1, 1024, 1024, 1] in [0, 1] for its normalized input [1, 1024, 1024, 3].
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let full = backbone(x)
        let half = backbone(idolResizeBilinear(x, height: x.dim(1) / 2, width: x.dim(2) / 2))
        var features = zip(full, half).map { a, b in concatenated([a, idolResizeBilinear(b, height: a.dim(1), width: a.dim(2))], axis: -1) }
        let top = features[3]
        features[3] = concatenated(features[0 ..< 3].map { idolResizeBilinear($0, height: top.dim(1), width: top.dim(2)) } + [top], axis: -1)
        let squeezed = squeezeModule[0](features[3])
        return sigmoid(decoder(x, features[0], features[1], features[2], squeezed))
    }

    /// `remove_background`: the picture's RGB bytes (`width` × `height`) → its alpha bytes, the same size.
    public func alpha(rgb: [UInt8], width: Int, height: Int) -> [UInt8] {
        let pixels = MLXArray(rgb.map { Float($0) / 255 }).reshaped([1, height, width, 3])
        let input = (idolResizeBilinear(pixels, height: Self.size, width: Self.size) - MLXArray(Self.mean)) / MLXArray(Self.std)
        let map = idolResizeBilinear(self(input), height: height, width: width)
        let bytes = (clip(map, min: 0, max: 1) * 255).asType(.uint8)
        eval(bytes)
        return bytes.asArray(UInt8.self)
    }
}
