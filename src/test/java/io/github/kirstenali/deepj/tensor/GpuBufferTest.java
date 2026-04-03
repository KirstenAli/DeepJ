package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GpuBufferTest {

    @Test
    void constructorSetsFieldsCorrectly() {
        GpuBuffer buf = new GpuBuffer(7, 3, 4, true);

        assertEquals(7, buf.id);
        assertEquals(3, buf.rows);
        assertEquals(4, buf.cols);
        assertTrue(buf.needsUpload);
        assertFalse(buf.cpuStale, "cpuStale should be false when needsUpload is true");
        assertFalse(buf.allocatedOnGpu, "allocatedOnGpu defaults to false");
    }

    @Test
    void constructorWithNeedsUploadFalse() {
        GpuBuffer buf = new GpuBuffer(0, 2, 5, false);

        assertFalse(buf.needsUpload);
        assertTrue(buf.cpuStale, "cpuStale should be true when needsUpload is false");
    }

    @Test
    void floatCountReturnsRowsTimesCols() {
        GpuBuffer buf = new GpuBuffer(0, 64, 128, true);
        assertEquals(64 * 128, buf.floatCount());
    }

    @Test
    void floatCountSingleElement() {
        GpuBuffer buf = new GpuBuffer(0, 1, 1, false);
        assertEquals(1, buf.floatCount());
    }
}

