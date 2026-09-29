import Foundation
import MLX
import TurboEngineCore

// turbo-engine: the native engine of Turbo MLX.
//
//   turbo-engine serve             the worker the app talks to (JSON lines on stdin/stdout)
//   turbo-engine verify <fixture>  checks a family's port against mflux's reference outputs; the
//                                  fixture names the family (see Engine/Fixtures/make_*_fixture.py)

setvbuf(stdout, nil, _IOLBF, 0)
signal(SIGPIPE, SIG_IGN)
// MLX (0.32) runs a clip's short 3D convolutions as 2D ones over all the frames of a tile, with
// Winograd buffers for as many frames as three quarters of the GPU's working set hold: a
// 5-second LTX clip peaked at 30 GB with Save memory, against 16 before. One frame per Winograd
// step brings it back to 18 GB and decodes faster still (14 s against 27 with MLX 0.31, 17
// without the limit). A still image is one frame: nothing changes for it. MLX reads the
// variable at each convolution; one set in the environment wins.
setenv("MLX_CONV_WINOGRAD_TILE_BATCH", "1", 0)

let arguments = CommandLine.arguments.dropFirst()
switch arguments.first ?? "serve" {
case "serve":
    Server().run()
case "verify":
    guard let path = arguments.dropFirst().first else {
        Emitter.shared.log("usage: turbo-engine verify <fixture-folder>")
        exit(2)
    }
    exit(Verify.run(fixture: URL(fileURLWithPath: path)) ? 0 : 1)
default:
    Emitter.shared.log("usage: turbo-engine [serve | verify <fixture-folder>]")
    exit(2)
}
