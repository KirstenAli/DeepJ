import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!
print("DEVICE", device.name)
print("APPLE7", device.supportsFamily(.apple7))
print("MAC2", device.supportsFamily(.mac2))
let source = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void probe(device const float* input [[buffer(0)]], device float* result [[buffer(1)]]) {
    simdgroup_float8x8 a, b, c(0.0f);
    simdgroup_load(a, input, 8);
    simdgroup_load(b, input + 64, 8);
    simdgroup_multiply_accumulate(c, a, b, c);
    simdgroup_store(c, result, 8);
}
"""

func run(_ pipeline: MTLComputePipelineState) {
    let input = device.makeBuffer(length: 128 * 4, options: .storageModeShared)!
    let data = input.contents().bindMemory(to: Float.self, capacity: 128)
    for index in 0..<128 { data[index] = Float(index % 13 - 6) * 0.125 }
    let buffer = device.makeBuffer(length: 64 * 4, options: .storageModeShared)!
    let command = device.makeCommandQueue()!.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(buffer, offset: 0, index: 1)
    encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    print("STATUS", command.status.rawValue, command.error as Any)
    let values = buffer.contents().bindMemory(to: Float.self, capacity: 64)
    check(data, values)
}

func check(_ input: UnsafeMutablePointer<Float>, _ result: UnsafeMutablePointer<Float>) {
    var maximum = Float(0)
    for row in 0..<8 {
        for column in 0..<8 {
            var expected = Float(0)
            for inner in 0..<8 { expected += input[row * 8 + inner] * input[64 + inner * 8 + column] }
            maximum = max(maximum, abs(expected - result[row * 8 + column]))
        }
    }
    print("DYNAMIC_MATRIX_MAX_ERROR", maximum)
    precondition(maximum < 1e-6)
}

do {
    let library = try device.makeLibrary(source: source, options: nil)
    run(try device.makeComputePipelineState(function: library.makeFunction(name: "probe")!))
} catch {
    print("MATRIX_UNAVAILABLE", error)
}
