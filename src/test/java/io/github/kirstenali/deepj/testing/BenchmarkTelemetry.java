package io.github.kirstenali.deepj.testing;

import java.lang.management.ManagementFactory;

final class BenchmarkTelemetry {
    private BenchmarkTelemetry() {}

    static String snapshot() {
        var heap = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage();
        long count = 0, milliseconds = 0;
        for (var collector : ManagementFactory.getGarbageCollectorMXBeans()) {
            count += Math.max(0, collector.getCollectionCount());
            milliseconds += Math.max(0, collector.getCollectionTime());
        }
        return "heapUsedBytes=" + heap.getUsed() + " heapCommittedBytes=" + heap.getCommitted()
                + " gcCount=" + count + " gcMillis=" + milliseconds;
    }
}
