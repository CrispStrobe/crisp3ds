#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { std::fprintf(stderr, "No Metal device\n"); return 2; }
        NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\n"
                            @"kernel void add(device const float* a [[buffer(0)]], "
                            @"device const float* b [[buffer(1)]], "
                            @"device float* out [[buffer(2)]], "
                            @"uint i [[thread_position_in_grid]]) { out[i] = a[i] + b[i]; }";
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) { std::fprintf(stderr, "Metal library: %s\n", error.localizedDescription.UTF8String); return 3; }
        id<MTLFunction> function = [library newFunctionWithName:@"add"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) { std::fprintf(stderr, "Metal pipeline: %s\n", error.localizedDescription.UTF8String); return 4; }
        constexpr size_t count = 4;
        const float a[count] = {1.0f, -2.0f, 3.5f, 0.0f};
        const float b[count] = {2.0f, 5.0f, -1.5f, 7.0f};
        const float expected[count] = {3.0f, 3.0f, 2.0f, 7.0f};
        id<MTLBuffer> aa = [device newBufferWithBytes:a length:sizeof(a) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bb = [device newBufferWithBytes:b length:sizeof(b) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [device newBufferWithLength:sizeof(a) options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!aa || !bb || !out || !queue || !command || !encoder) return 5;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:aa offset:0 atIndex:0];
        [encoder setBuffer:bb offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(count, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "Metal command failed: %s\n", command.error.localizedDescription.UTF8String);
            return 6;
        }
        const float *actual = static_cast<const float *>(out.contents);
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 1e-6f) {
                std::fprintf(stderr, "Mismatch at %zu: %.8g vs %.8g\n", i, actual[i], expected[i]);
                return 7;
            }
        }
        std::printf("Metal compute smoke passed on %s: 4/4 values\n", device.name.UTF8String);
        return 0;
    }
}
