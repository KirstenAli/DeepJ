import Foundation
import Metal

let devices = MTLCopyAllDevices()
print("Metal device count: \(devices.count)")
for device in devices {
    print("Metal device: \(device.name), unifiedMemory=\(device.hasUnifiedMemory)")
    print("Supports Apple GPU family 1: \(device.supportsFamily(.apple1))")
}

guard let output = ProcessInfo.processInfo.environment["GITHUB_OUTPUT"] else {
    fatalError("Missing GitHub Actions step output path")
}
let file = try FileHandle(forWritingTo: URL(fileURLWithPath: output))
try file.seekToEnd()
try file.write(contentsOf: Data("available=\(!devices.isEmpty)\n".utf8))
try file.close()
