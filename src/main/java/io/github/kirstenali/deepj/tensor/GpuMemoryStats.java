package io.github.kirstenali.deepj.tensor;

public record GpuMemoryStats(int bufferCount, long allocatedBytes) {

    public static GpuMemoryStats empty() {
        return new GpuMemoryStats(0, 0L);
    }
}
