import Foundation
import MLX
import TurboEngineCore

// turbo-engine: the native engine of Turbo MLX.
//
//   turbo-engine serve             the worker the app talks to (JSON lines on stdin/stdout)
//   turbo-engine verify <fixture>  checks the FLUX.2 Klein port against mflux's reference outputs
//                                  (see Engine/Fixtures/make_klein_fixture.py)

setvbuf(stdout, nil, _IOLBF, 0)
signal(SIGPIPE, SIG_IGN)

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
