package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

final class GraphTransferDispatchTest {
    @Test
    void implicitFloat32ZerosKeepTheOriginalEntryPoint() {
        for (int count : new int[]{1, 2, 7, 32}) {
            var runtime = new RecordingRuntime();
            GraphTransfers.upload(runtime, new int[]{7, count}, null);
            assertEquals("legacy", runtime.route);
            assertEquals(7, runtime.id);
            assertEquals(count, runtime.elements);
            assertNull(runtime.dtype);
        }
    }

    @Test
    void explicitlyTypedZerosKeepTheirDtypeEntryPointForEveryStorageFormat() {
        for (DType dtype : DType.values()) {
            var runtime = new RecordingRuntime();
            GraphTransfers.upload(runtime, new int[]{7, 3, dtype.storageCode()}, null);
            assertEquals("typed", runtime.route);
            assertEquals(7, runtime.id);
            assertEquals(3, runtime.elements);
            assertEquals(dtype, runtime.dtype);
        }
    }

    @Test
    void uploadsForwardOriginalBitPatternsAndPayloadReferences() {
        var runtime = new RecordingRuntime();
        float[] floats = {Float.intBitsToFloat(0x7fa01234), -0.0f, Float.POSITIVE_INFINITY};
        GraphTransfers.upload(runtime, new int[]{7}, floats);
        assertEquals("float", runtime.route);
        assertSame(floats, runtime.payload);
        assertEquals(0x7fa01234, Float.floatToRawIntBits(floats[0]));
        assertEquals(0x80000000, Float.floatToRawIntBits(floats[1]));
        short[] compact = {(short) 0x7e01, (short) 0x8000, (short) 0x7c00};
        GraphTransfers.upload(runtime, new int[]{8}, compact);
        assertEquals("compact", runtime.route);
        assertSame(compact, runtime.payload);
        assertArrayEquals(new short[]{(short) 0x7e01, (short) 0x8000, (short) 0x7c00}, compact);
        assertEquals(8, runtime.id);
    }

    @Test
    void invalidMetadataAndRuntimeCountValidationRemainVisible() {
        var runtime = new RecordingRuntime();
        assertThrows(NullPointerException.class, () -> GraphTransfers.upload(runtime, null, null));
        assertThrows(ArrayIndexOutOfBoundsException.class,
                () -> GraphTransfers.upload(runtime, new int[]{7}, null));
        assertThrows(IllegalArgumentException.class, () -> GraphTransfers.upload(runtime, new int[]{7, 0}, null));
        assertThrows(IllegalArgumentException.class, () -> GraphTransfers.upload(runtime, new int[]{7, -1}, null));
        assertThrows(IllegalArgumentException.class, () -> GraphTransfers.upload(runtime, new int[]{-1, 2}, null));
        assertThrows(IllegalArgumentException.class,
                () -> GraphTransfers.upload(runtime, new int[]{7, 2, -1}, null));
        assertEquals("", runtime.route);
    }

    private static final class RecordingRuntime implements GpuRuntime {
        private String route = "";
        private int id, elements;
        private Object payload;
        private DType dtype;

        @Override public void allocBuffers(int[] ids, int[] sizes, int count) { fail("No allocation expected"); }
        @Override public void releaseBuffers(int[] ids, int count) { fail("No release expected"); }
        @Override public void flushOps(int[] commands, int length) { fail("No computation expected"); }
        @Override public void downloadBuffer(int id, float[] values) { fail("No download expected"); }

        @Override public void uploadBuffer(int id, float[] data) { upload(id, data, "float"); }
        @Override public void uploadBuffer16(int id, short[] data) { upload(id, data, "compact"); }

        private void upload(int buffer, Object data, String selected) {
            id = buffer;
            payload = data;
            route = selected;
        }

        @Override public void zeroBuffer(int id, int count) { zero(id, count, null); }
        @Override public void zeroBuffer(int id, int count, DType dtype) { zero(id, count, dtype); }

        private void zero(int buffer, int count, DType type) {
            if (buffer < 0 || count < 1) throw new IllegalArgumentException("Invalid zero request");
            id = buffer;
            elements = count;
            dtype = type;
            route = type == null ? "legacy" : "typed";
        }
    }
}
