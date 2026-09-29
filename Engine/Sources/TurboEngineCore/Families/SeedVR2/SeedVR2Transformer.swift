import Foundation
import MLX
import MLXNN

// SeedVR2's transformer, after mflux's `seedvr2_transformer/*`: 32 blocks over the video tokens
// (the noisy latent next to the picture's latent, in 2 × 2 patches) and 58 fixed text tokens, the
// video attending within windows (shifted every other block) together with the whole text. The
// modules are named as the original checkpoint (numz/SeedVR2_comfyUI) names its tensors, so it
// loads unchanged: `vid` and `txt` weights in the first 10 blocks, shared `all` ones after. The
// checkpoint is float16 and mflux keeps it so; the latents are float32, so the video stream runs
// in float32 as in mflux, which this port follows by using the same operations without casts.

/// `RMSNorm`: `mx.fast.rms_norm` with a learned weight.
final class SeedVR2RMSNorm: Module {
    @ParameterInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// `TransformerBlock._rms_norm`: RMS over the last axis with a float32 weight of ones.
private func plainRMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    MLXFast.rmsNorm(x, weight: MLXArray.ones([x.shape[x.ndim - 1]]), eps: eps)
}

/// A layer of the video stream and one of the text stream, or one both share (`all`).
final class SeedVR2LinearStreams: Module {
    @ModuleInfo(key: "vid") var vid: Linear?
    @ModuleInfo(key: "txt") var txt: Linear?
    @ModuleInfo(key: "all") var all: Linear?

    init(vidDim: Int, txtDim: Int, outVid: Int, outTxt: Int, bias: Bool, shared: Bool) {
        if shared {
            _all.wrappedValue = Linear(vidDim, outVid, bias: bias)
        } else {
            _vid.wrappedValue = Linear(vidDim, outVid, bias: bias)
            _txt.wrappedValue = Linear(txtDim, outTxt, bias: bias)
        }
        super.init()
    }

    var video: Linear { all ?? vid! }
    var text: Linear { all ?? txt! }
}

final class SeedVR2NormStreams: Module {
    @ModuleInfo(key: "vid") var vid: SeedVR2RMSNorm?
    @ModuleInfo(key: "txt") var txt: SeedVR2RMSNorm?
    @ModuleInfo(key: "all") var all: SeedVR2RMSNorm?

    init(dimensions: Int, eps: Float, shared: Bool) {
        if shared {
            _all.wrappedValue = SeedVR2RMSNorm(dimensions: dimensions, eps: eps)
        } else {
            _vid.wrappedValue = SeedVR2RMSNorm(dimensions: dimensions, eps: eps)
            _txt.wrappedValue = SeedVR2RMSNorm(dimensions: dimensions, eps: eps)
        }
        super.init()
    }

    var video: SeedVR2RMSNorm { all ?? vid! }
    var text: SeedVR2RMSNorm { all ?? txt! }
}

/// The rotary frequencies each block stores (`attn.rope.rope.freqs`, float16): mflux reads them
/// from the checkpoint rather than computing them.
final class SeedVR2RoPE: Module {
    @ParameterInfo var freqs: MLXArray

    init(dim: Int, theta: Double = 10000) {
        let freqDim = dim / 3
        let values = Swift.stride(from: 0, to: freqDim, by: 2).prefix(freqDim / 2).map { i in
            Float(1 / pow(theta, Double(i) / Double(freqDim)))
        }
        _freqs.wrappedValue = MLXArray(values)
        super.init()
    }
}

final class SeedVR2RoPEHolder: Module {
    @ModuleInfo(key: "rope") var rope: SeedVR2RoPE

    init(dim: Int) {
        _rope.wrappedValue = SeedVR2RoPE(dim: dim)
        super.init()
    }
}

/// The windows the video tokens attend within, as `WindowPartitioner` cuts them: the partition
/// (token indices, window after window), its inverse, and each window's extent. Windows of the same
/// length follow each other so that each run of them is one batched attention; `originalIndex`
/// keeps mflux's order, which the text's mean over windows is taken in.
public struct SeedVR2Windows {
    public struct Window {
        public let t: Range<Int>
        public let h: Range<Int>
        public let w: Range<Int>
        public var length: Int { t.count * h.count * w.count }
    }

    public let windows: [Window]
    /// For each window here, its position in mflux's order.
    public let originalIndex: [Int]
    public let forward: MLXArray
    public let reverse: MLXArray
    /// Runs of windows of equal length: the first window, how many, their length.
    public let runs: [(first: Int, count: Int, length: Int)]

    /// `WindowPartitioner._make_windows` for a (t, h, w) grid and (nt, nh, nw) windows per axis.
    public static func mfluxWindows(t: Int, h: Int, w: Int, numWindows: (t: Int, h: Int, w: Int), shift: Bool) -> [Window] {
        let scale = (Double(45 * 80) / Double(h * w)).squareRoot()
        let resizedH = (Double(h) * scale).rounded(.toNearestOrEven)
        let resizedW = (Double(w) * scale).rounded(.toNearestOrEven)
        let wh = Int((resizedH / Double(numWindows.h)).rounded(.up))
        let ww = Int((resizedW / Double(numWindows.w)).rounded(.up))
        let wt = Int((Double(min(t, 30)) / Double(numWindows.t)).rounded(.up))

        let st: Double, sh: Double, sw: Double
        let nt: Int, nh: Int, nw: Int
        if shift {
            st = wt < t ? 0.5 : 0
            sh = wh < h ? 0.5 : 0
            sw = ww < w ? 0.5 : 0
            nt = st > 0 ? Int(((Double(t) - st) / Double(wt)).rounded(.up)) + 1 : 1
            nh = sh > 0 ? Int(((Double(h) - sh) / Double(wh)).rounded(.up)) + 1 : 1
            nw = sw > 0 ? Int(((Double(w) - sw) / Double(ww)).rounded(.up)) + 1 : 1
        } else {
            (st, sh, sw) = (0, 0, 0)
            nt = Int((Double(t) / Double(wt)).rounded(.up))
            nh = Int((Double(h) / Double(wh)).rounded(.up))
            nw = Int((Double(w) / Double(ww)).rounded(.up))
        }
        // Python's int() truncates toward zero, as Swift's does.
        func bounds(_ i: Int, _ s: Double, _ size: Int, _ extent: Int) -> Range<Int>? {
            let start = max(Int((Double(i) - s) * Double(size)), 0)
            let end = min(Int((Double(i) - s + 1) * Double(size)), extent)
            return end > start ? start ..< end : nil
        }
        var windows: [Window] = []
        for iw in 0 ..< nw {
            guard let wRange = bounds(iw, sw, ww, w) else { continue }
            for ih in 0 ..< nh {
                guard let hRange = bounds(ih, sh, wh, h) else { continue }
                for it in 0 ..< nt {
                    guard let tRange = bounds(it, st, wt, t) else { continue }
                    windows.append(Window(t: tRange, h: hRange, w: wRange))
                }
            }
        }
        return windows
    }

    public init(t: Int, h: Int, w: Int, numWindows: (t: Int, h: Int, w: Int), shift: Bool) {
        let mflux = Self.mfluxWindows(t: t, h: h, w: w, numWindows: numWindows, shift: shift)
        // Grouped by length, mflux's order kept within a length.
        let order = mflux.indices.sorted { a, b in
            mflux[a].length != mflux[b].length ? mflux[a].length > mflux[b].length : a < b
        }
        windows = order.map { mflux[$0] }
        originalIndex = order

        var indices: [Int32] = []
        indices.reserveCapacity(t * h * w)
        for window in windows {
            for tt in window.t {
                for yy in window.h {
                    for xx in window.w {
                        indices.append(Int32((tt * h + yy) * w + xx))
                    }
                }
            }
        }
        var inverse = [Int32](repeating: 0, count: indices.count)
        for (position, index) in indices.enumerated() { inverse[Int(index)] = Int32(position) }
        forward = MLXArray(indices)
        reverse = MLXArray(inverse)

        var runs: [(first: Int, count: Int, length: Int)] = []
        for (index, window) in windows.enumerated() {
            if let last = runs.last, last.length == window.length {
                runs[runs.count - 1].count += 1
            } else {
                runs.append((index, 1, window.length))
            }
        }
        self.runs = runs
    }

    /// Each partitioned token's (t, y, x) position for the rotary embedding: local to its window,
    /// the time axis starting after the text (`vid_freqs_full[txt_len:txt_len + f, :h, :w]`).
    func positions(textLength: Int) -> (t: MLXArray, y: MLXArray, x: MLXArray) {
        var pt: [Float] = [], py: [Float] = [], px: [Float] = []
        for window in windows {
            for tt in 0 ..< window.t.count {
                for yy in 0 ..< window.h.count {
                    for xx in 0 ..< window.w.count {
                        pt.append(Float(textLength + tt))
                        py.append(Float(yy))
                        px.append(Float(xx))
                    }
                }
            }
        }
        return (MLXArray(pt), MLXArray(py), MLXArray(px))
    }
}

/// `RoPEModule._apply_rotary_emb`: the first `angles.shape[-1]` features rotated in pairs, in
/// float32, then back to the input's dtype. `x` is [N, heads, D], `angles` [N, 1, R].
func seedVR2Rotate(_ x: MLXArray, angles: MLXArray) -> MLXArray {
    let rotated = angles.shape[angles.ndim - 1]
    let dtype = x.dtype
    let middle = x[.ellipsis, 0 ..< rotated].asType(.float32)
    let pairs = middle.reshaped(Array(middle.shape.dropLast()) + [rotated / 2, 2])
    let half = stacked([-pairs[.ellipsis, 1], pairs[.ellipsis, 0]], axis: -1).reshaped(middle.shape)
    let angles = angles.asType(.float32)
    let transformed = (middle * cos(angles) + half * sin(angles)).asType(dtype)
    let rest = x.shape[x.ndim - 1] - rotated
    return rest > 0 ? concatenated([transformed, x[.ellipsis, rotated...]], axis: -1) : transformed
}

/// `MMAttention`: queries, keys and values of both streams, the video's cut into windows, each
/// window attending over its tokens and the whole text; the text's output is the mean over the
/// windows.
final class SeedVR2Attention: Module {
    @ModuleInfo(key: "proj_qkv") var projQKV: SeedVR2LinearStreams
    @ModuleInfo(key: "proj_out") var projOut: SeedVR2LinearStreams
    @ModuleInfo(key: "norm_q") var normQ: SeedVR2NormStreams
    @ModuleInfo(key: "norm_k") var normK: SeedVR2NormStreams
    @ModuleInfo(key: "rope") var rope: SeedVR2RoPEHolder

    let heads: Int
    let headDim: Int
    let scale: Float

    init(vidDim: Int, txtDim: Int, heads: Int, headDim: Int, eps: Float, ropeDim: Int, shared: Bool) {
        self.heads = heads
        self.headDim = headDim
        scale = pow(Float(headDim), -0.5)
        let inner = heads * headDim
        _projQKV.wrappedValue = SeedVR2LinearStreams(vidDim: vidDim, txtDim: txtDim, outVid: 3 * inner, outTxt: 3 * inner, bias: false, shared: shared)
        _projOut.wrappedValue = SeedVR2LinearStreams(vidDim: inner, txtDim: inner, outVid: vidDim, outTxt: txtDim, bias: true, shared: shared)
        _normQ.wrappedValue = SeedVR2NormStreams(dimensions: headDim, eps: eps, shared: shared)
        _normK.wrappedValue = SeedVR2NormStreams(dimensions: headDim, eps: eps, shared: shared)
        _rope.wrappedValue = SeedVR2RoPEHolder(dim: ropeDim)
        super.init()
    }

    /// The rotary angles of the partitioned video tokens [L, 1, R] and of the text [T, 1, R]
    /// (`_apply_mm_rope_3d` with "lang" frequencies: positions times frequencies, each repeated
    /// twice, the three axes one after the other; the text at the same position on all three).
    func angles(windows: SeedVR2Windows, positions: (t: MLXArray, y: MLXArray, x: MLXArray), textLength: Int) -> (vid: MLXArray, txt: MLXArray) {
        let freqs = repeated(rope.rope.freqs.asType(.float32), count: 2, axis: -1)
        let vid = concatenated([outer(positions.t, freqs), outer(positions.y, freqs), outer(positions.x, freqs)], axis: -1)
        let text = outer(MLXArray(0 ..< textLength).asType(.float32), freqs)
        let txt = tiled(text, repetitions: [1, 3])
        return (expandedDimensions(vid, axis: 1), expandedDimensions(txt, axis: 1))
    }

    func callAsFunction(vid: MLXArray, txt: MLXArray, windows: SeedVR2Windows, positions: (t: MLXArray, y: MLXArray, x: MLXArray)) -> (MLXArray, MLXArray) {
        let (b, l, bt, lt) = (vid.shape[0], vid.shape[1], txt.shape[0], txt.shape[1])
        let qkvVid = projQKV.video(vid.reshaped([-1, vid.shape[2]])).reshaped([-1, 3, heads, headDim]).take(windows.forward, axis: 0)
        let qkvTxt = projQKV.text(txt.reshaped([-1, txt.shape[2]])).reshaped([-1, 3, heads, headDim])

        var qVid = normQ.video(qkvVid[0..., 0])
        var kVid = normK.video(qkvVid[0..., 1])
        let vVid = qkvVid[0..., 2]
        var qTxt = normQ.text(qkvTxt[0..., 0])
        var kTxt = normK.text(qkvTxt[0..., 1])
        let vTxt = qkvTxt[0..., 2]

        let rotary = angles(windows: windows, positions: positions, textLength: lt)
        qVid = seedVR2Rotate(qVid, angles: rotary.vid)
        kVid = seedVR2Rotate(kVid, angles: rotary.vid)
        qTxt = seedVR2Rotate(qTxt, angles: rotary.txt)
        kTxt = seedVR2Rotate(kTxt, angles: rotary.txt)

        // Each run of equal windows as one batch: [n, heads, len + text, D].
        var vidParts: [MLXArray] = []
        var txtParts: [MLXArray] = []
        var offset = 0
        for run in windows.runs {
            let span = offset ..< (offset + run.count * run.length)
            offset += run.count * run.length
            func joined(_ video: MLXArray, _ text: MLXArray) -> MLXArray {
                let v = video[span].reshaped([run.count, run.length, heads, headDim])
                let t = broadcast(expandedDimensions(text, axis: 0), to: [run.count, lt, heads, headDim])
                return concatenated([v, t], axis: 1).transposed(0, 2, 1, 3)
            }
            let out = MLXFast.scaledDotProductAttention(
                queries: joined(qVid, qTxt), keys: joined(kVid, kTxt), values: joined(vVid, vTxt), scale: scale, mask: nil
            ).transposed(0, 2, 1, 3)
            vidParts.append(out[0..., 0 ..< run.length].reshaped([run.count * run.length, heads * headDim]))
            txtParts.append(out[0..., run.length...].reshaped([run.count, lt, heads * headDim]))
        }
        let vidOut = concatenated(vidParts, axis: 0).take(windows.reverse, axis: 0)
        // The text's outputs in mflux's window order, averaged.
        var inMfluxOrder = [Int](repeating: 0, count: windows.windows.count)
        for (position, original) in windows.originalIndex.enumerated() { inMfluxOrder[original] = position }
        let txtOut = concatenated(txtParts, axis: 0).take(MLXArray(inMfluxOrder.map { Int32($0) }), axis: 0).mean(axis: 0)

        return (projOut.video(vidOut).reshaped([b, l, -1]), projOut.text(txtOut).reshaped([bt, lt, -1]))
    }
}

/// `SwiGLUMLP`: SiLU of the gate times the input projection, projected back.
final class SeedVR2SwiGLU: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_in_gate") var projInGate: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(dim: Int, expandRatio: Int, multipleOf: Int = 256) {
        var hidden = Int(Double(2 * dim * expandRatio) / 3)
        hidden = multipleOf * ((hidden + multipleOf - 1) / multipleOf)
        _projIn.wrappedValue = Linear(dim, hidden, bias: false)
        _projInGate.wrappedValue = Linear(dim, hidden, bias: false)
        _projOut.wrappedValue = Linear(hidden, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = silu(projInGate(x))
        return projOut(gate * projIn(x))
    }
}

final class SeedVR2MLPStreams: Module {
    @ModuleInfo(key: "vid") var vid: SeedVR2SwiGLU?
    @ModuleInfo(key: "txt") var txt: SeedVR2SwiGLU?
    @ModuleInfo(key: "all") var all: SeedVR2SwiGLU?

    init(vidDim: Int, txtDim: Int, expandRatio: Int, shared: Bool, isLastLayer: Bool) {
        if shared {
            _all.wrappedValue = SeedVR2SwiGLU(dim: vidDim, expandRatio: expandRatio)
        } else {
            _vid.wrappedValue = SeedVR2SwiGLU(dim: vidDim, expandRatio: expandRatio)
            if !isLastLayer { _txt.wrappedValue = SeedVR2SwiGLU(dim: txtDim, expandRatio: expandRatio) }
        }
        super.init()
    }
}

/// The learned offsets of one stream's modulation.
final class SeedVR2AdaParams: Module {
    @ParameterInfo(key: "attn_shift") var attnShift: MLXArray
    @ParameterInfo(key: "attn_scale") var attnScale: MLXArray
    @ParameterInfo(key: "attn_gate") var attnGate: MLXArray
    @ParameterInfo(key: "mlp_shift") var mlpShift: MLXArray
    @ParameterInfo(key: "mlp_scale") var mlpScale: MLXArray
    @ParameterInfo(key: "mlp_gate") var mlpGate: MLXArray

    init(dim: Int) {
        _attnShift.wrappedValue = MLXArray.zeros([dim])
        _attnScale.wrappedValue = MLXArray.ones([dim])
        _attnGate.wrappedValue = MLXArray.zeros([dim])
        _mlpShift.wrappedValue = MLXArray.zeros([dim])
        _mlpScale.wrappedValue = MLXArray.ones([dim])
        _mlpGate.wrappedValue = MLXArray.zeros([dim])
        super.init()
    }
}

/// `AdaModulation`: shift and scale in, gate out, from the timestep's embedding plus the learned
/// offsets; the last block leaves the text as it is.
final class SeedVR2Ada: Module {
    @ModuleInfo(key: "vid") var vid: SeedVR2AdaParams?
    @ModuleInfo(key: "txt") var txt: SeedVR2AdaParams?
    @ModuleInfo(key: "all") var all: SeedVR2AdaParams?

    let isLastLayer: Bool

    enum Layer { case attn, mlp }

    init(dim: Int, shared: Bool, isLastLayer: Bool) {
        self.isLastLayer = isLastLayer
        if shared {
            _all.wrappedValue = SeedVR2AdaParams(dim: dim)
        } else {
            _vid.wrappedValue = SeedVR2AdaParams(dim: dim)
            if !isLastLayer { _txt.wrappedValue = SeedVR2AdaParams(dim: dim) }
        }
        super.init()
    }

    func modulateVid(_ hidden: MLXArray, emb: MLXArray, layer: Layer, input: Bool) -> MLXArray {
        Self.apply(hidden, emb: emb, params: all ?? vid!, layer: layer, input: input)
    }

    func modulateTxt(_ hidden: MLXArray, emb: MLXArray, layer: Layer, input: Bool) -> MLXArray {
        if isLastLayer { return hidden }
        return Self.apply(hidden, emb: emb, params: all ?? txt!, layer: layer, input: input)
    }

    /// `emb` is [B, dim, 2, 3]: (attention, MLP) × (shift, scale, gate).
    static func apply(_ hidden: MLXArray, emb: MLXArray, params: SeedVR2AdaParams, layer: Layer, input: Bool) -> MLXArray {
        let mod = emb[0..., 0..., layer == .attn ? 0 : 1, 0...]
        if input {
            let shift = expandedDimensions(mod[.ellipsis, 0], axis: 1) + (layer == .attn ? params.attnShift : params.mlpShift)
            let scale = expandedDimensions(mod[.ellipsis, 1], axis: 1) + (layer == .attn ? params.attnScale : params.mlpScale)
            return hidden * scale + shift
        }
        let gate = expandedDimensions(mod[.ellipsis, 2], axis: 1) + (layer == .attn ? params.attnGate : params.mlpGate)
        return hidden * gate
    }
}

final class SeedVR2Block: Module {
    @ModuleInfo(key: "attn") var attn: SeedVR2Attention
    @ModuleInfo(key: "mlp") var mlp: SeedVR2MLPStreams
    @ModuleInfo(key: "ada") var ada: SeedVR2Ada

    let isLastLayer: Bool
    let shift: Bool
    let eps: Float

    init(config: SeedVR2Config, shared: Bool, isLastLayer: Bool, shift: Bool) {
        self.isLastLayer = isLastLayer
        self.shift = shift
        eps = config.normEps
        _attn.wrappedValue = SeedVR2Attention(vidDim: config.vidDim, txtDim: config.vidDim, heads: config.heads, headDim: config.headDim,
                                              eps: config.normEps, ropeDim: config.ropeDim, shared: shared)
        _mlp.wrappedValue = SeedVR2MLPStreams(vidDim: config.vidDim, txtDim: config.vidDim, expandRatio: config.expandRatio,
                                              shared: shared, isLastLayer: isLastLayer)
        _ada.wrappedValue = SeedVR2Ada(dim: config.vidDim, shared: shared, isLastLayer: isLastLayer)
        super.init()
    }

    func callAsFunction(vid: MLXArray, txt: MLXArray, emb: MLXArray, windows: SeedVR2Windows,
                        positions: (t: MLXArray, y: MLXArray, x: MLXArray)) -> (MLXArray, MLXArray) {
        var vidAttn = ada.modulateVid(plainRMSNorm(vid, eps: eps), emb: emb, layer: .attn, input: true)
        var txtAttn = ada.modulateTxt(plainRMSNorm(txt, eps: eps), emb: emb, layer: .attn, input: true)
        (vidAttn, txtAttn) = attn(vid: vidAttn, txt: txtAttn, windows: windows, positions: positions)
        vidAttn = ada.modulateVid(vidAttn, emb: emb, layer: .attn, input: false)
        txtAttn = ada.modulateTxt(txtAttn, emb: emb, layer: .attn, input: false)

        var vid = vid + vidAttn
        var txt = txt
        if !isLastLayer { txt = txt + txtAttn }

        let vidMLP = ada.modulateVid(plainRMSNorm(vid, eps: eps), emb: emb, layer: .mlp, input: true)
        // The last block's text output is never used: mflux computes it and drops it.
        vid = vid + ada.modulateVid((mlp.all ?? mlp.vid!)(vidMLP), emb: emb, layer: .mlp, input: false)
        if !isLastLayer {
            let txtMLP = ada.modulateTxt(plainRMSNorm(txt, eps: eps), emb: emb, layer: .mlp, input: true)
            txt = txt + ada.modulateTxt((mlp.all ?? mlp.txt!)(txtMLP), emb: emb, layer: .mlp, input: false)
        }
        return (vid, txt)
    }
}

/// `TimeEmbedding`: sinusoidal features (sin first) through three linears.
final class SeedVR2TimeEmbedding: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_hid") var projHid: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    let sinusoidalDim: Int

    init(sinusoidalDim: Int = 256, hiddenDim: Int, outputDim: Int) {
        self.sinusoidalDim = sinusoidalDim
        _projIn.wrappedValue = Linear(sinusoidalDim, hiddenDim)
        _projHid.wrappedValue = Linear(hiddenDim, hiddenDim)
        _projOut.wrappedValue = Linear(hiddenDim, outputDim)
        super.init()
    }

    func callAsFunction(_ timestep: Float) -> MLXArray {
        let half = sinusoidalDim / 2
        let freqs = exp(MLXArray(0 ..< half).asType(.float32) * Float(-log(10000.0) / Double(half)))
        let args = expandedDimensions(MLXArray([timestep]), axis: 1) * freqs
        var emb = concatenated([sin(args), cos(args)], axis: -1)
        emb = silu(projIn(emb))
        emb = silu(projHid(emb))
        return projOut(emb)
    }
}

final class SeedVR2Projection: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ inputs: Int, _ outputs: Int) {
        _proj.wrappedValue = Linear(inputs, outputs)
        super.init()
    }
}

final class SeedVR2OutAda: Module {
    @ParameterInfo(key: "out_shift") var outShift: MLXArray
    @ParameterInfo(key: "out_scale") var outScale: MLXArray

    init(dim: Int) {
        _outShift.wrappedValue = MLXArray.zeros([dim])
        _outScale.wrappedValue = MLXArray.ones([dim])
        super.init()
    }
}

public final class SeedVR2Transformer: Module {
    @ModuleInfo(key: "vid_in") var vidIn: SeedVR2Projection
    @ModuleInfo(key: "txt_in") var txtIn: Linear
    @ModuleInfo(key: "emb_in") var embIn: SeedVR2TimeEmbedding
    @ModuleInfo(key: "blocks") var blocks: [SeedVR2Block]
    @ModuleInfo(key: "vid_out_norm") var vidOutNorm: SeedVR2RMSNorm
    @ModuleInfo(key: "vid_out_ada") var vidOutAda: SeedVR2OutAda
    @ModuleInfo(key: "vid_out") var vidOut: SeedVR2Projection

    let config: SeedVR2Config

    public init(config: SeedVR2Config) {
        self.config = config
        let patch = 1 * 2 * 2
        _vidIn.wrappedValue = SeedVR2Projection(config.vidInChannels * patch, config.vidDim)
        _txtIn.wrappedValue = Linear(config.txtInDim, config.vidDim)
        _embIn.wrappedValue = SeedVR2TimeEmbedding(hiddenDim: config.vidDim, outputDim: config.embDim)
        _blocks.wrappedValue = (0 ..< config.numLayers).map { i in
            SeedVR2Block(config: config, shared: i >= config.mmLayers, isLastLayer: i == config.numLayers - 1, shift: i % 2 == 1)
        }
        _vidOutNorm.wrappedValue = SeedVR2RMSNorm(dimensions: config.vidDim, eps: config.normEps)
        _vidOutAda.wrappedValue = SeedVR2OutAda(dim: config.vidDim)
        _vidOut.wrappedValue = SeedVR2Projection(config.vidDim, config.vidOutChannels * patch)
        super.init()
    }

    /// `vid` [1, 33, 1, H, W] (the noise, the picture's latent, a mask of ones; channels first as
    /// in mflux), `txt` [1, T, 5120] → the predicted flow [1, 16, 1, H, W].
    public func callAsFunction(vid: MLXArray, txt: MLXArray, timestep: Float) -> MLXArray {
        // Never cancelled, so it never throws.
        try! forward(vid: vid, txt: txt, timestep: timestep)
    }

    /// The same pass, evaluated block by block and stopped between blocks when `isCancelled` says
    /// so: the upscale's only step lasts a minute at 4096 × 4096.
    public func forward(vid: MLXArray, txt: MLXArray, timestep: Float, isCancelled: () -> Bool = { false }) throws -> MLXArray {
        var txt = txtIn(txt)
        let (b, c, frames, height, width) = (vid.shape[0], vid.shape[1], vid.shape[2], vid.shape[3], vid.shape[4])
        let (tp, hp, wp) = (frames, height / 2, width / 2)
        var vid = vid.reshaped([b, c, tp, 1, hp, 2, wp, 2]).transposed(0, 2, 4, 6, 3, 5, 7, 1).reshaped([b, tp, hp, wp, 4 * c])
        vid = vidIn.proj(vid).reshaped([b, -1, config.vidDim])
        let emb = embIn(timestep).reshaped([-1, config.vidDim, 2, 3])

        let textLength = txt.shape[1]
        var partitions: [Bool: (SeedVR2Windows, (t: MLXArray, y: MLXArray, x: MLXArray))] = [:]
        for shift in [false, true] {
            let windows = SeedVR2Windows(t: tp, h: hp, w: wp, numWindows: config.window, shift: shift)
            partitions[shift] = (windows, windows.positions(textLength: textLength))
        }
        for block in blocks {
            let (windows, positions) = partitions[block.shift]!
            (vid, txt) = block(vid: vid, txt: txt, emb: emb, windows: windows, positions: positions)
            eval(vid, txt)
            if isCancelled() { throw GenerationError.cancelled }
        }

        vid = vidOutNorm(vid)
        let shiftA = expandedDimensions(emb[0..., 0..., 0, 0], axis: 1)
        let scaleA = expandedDimensions(emb[0..., 0..., 0, 1], axis: 1)
        vid = vid * (scaleA + vidOutAda.outScale) + (shiftA + vidOutAda.outShift)

        vid = vidOut.proj(vid)
        let channels = vid.shape[vid.ndim - 1] / 4
        return vid.reshaped([b, tp, hp, wp, 1, 2, 2, channels]).transposed(0, 7, 1, 4, 2, 5, 3, 6)
            .reshaped([b, channels, tp, hp * 2, wp * 2])
    }
}
