// A minimal original RealityKit Object Capture probe, not Apple sample code.
import Foundation
import RealityKit

enum ProbeFailure: Error, CustomStringConvertible {
    case usage(String)
    case unsupported
    case request(String)

    var description: String {
        switch self {
        case .usage(let message): return "usage: \(message)"
        case .unsupported: return "PhotogrammetrySession.isSupported is false on this Mac"
        case .request(let message): return "photogrammetry request failed: \(message)"
        }
    }
}

@main
struct Probe {
    static func main() async {
        do {
            try await run(Array(CommandLine.arguments.dropFirst()))
        } catch {
            fputs("ERROR \(error)\n", stderr)
            exit(1)
        }
    }

    static func run(_ args: [String]) async throws {
        if args == ["--check-support"] {
            let supported = PhotogrammetrySession.isSupported
            print("{\"schema\":\"apple_photogrammetry_support_v1\",\"supported\":\(supported)}")
            if !supported { throw ProbeFailure.unsupported }
            return
        }
        guard args.count == 4, args[0] == "--images", args[2] == "--output" else {
            throw ProbeFailure.usage("--check-support OR --images FOLDER --output FRESH.usdz")
        }
        guard PhotogrammetrySession.isSupported else { throw ProbeFailure.unsupported }
        let images = URL(fileURLWithPath: args[1], isDirectory: true)
        let output = URL(fileURLWithPath: args[3])
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: images.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw ProbeFailure.usage("input folder does not exist")
        }
        guard output.pathExtension.lowercased() == "usdz",
              !FileManager.default.fileExists(atPath: output.path) else {
            throw ProbeFailure.usage("output must be a fresh .usdz path")
        }
        let request = PhotogrammetrySession.Request.modelFile(url: output, detail: .preview)
        let session = try PhotogrammetrySession(input: images,
                                               configuration: PhotogrammetrySession.Configuration())
        try session.process(requests: [request])
        var requestFinished = false
        for try await event in session.outputs {
            switch event {
            case .requestComplete(_, let result):
                if case .modelFile(let file) = result {
                    guard file.standardizedFileURL == output.standardizedFileURL else {
                        throw ProbeFailure.request("unexpected output URL")
                    }
                    requestFinished = true
                    print("REQUEST_COMPLETE \(file.path)")
                }
            case .requestError(_, let error):
                throw ProbeFailure.request(String(describing: error))
            case .processingCancelled:
                throw ProbeFailure.request("cancelled")
            case .processingComplete:
                guard requestFinished, FileManager.default.fileExists(atPath: output.path) else {
                    throw ProbeFailure.request("processing ended without USDZ")
                }
                print("PROCESSING_COMPLETE \(output.path)")
                return
            case .requestProgress(_, let progress):
                print(String(format: "PROGRESS %.3f", progress))
            default:
                break
            }
        }
        throw ProbeFailure.request("output stream ended before completion")
    }
}
