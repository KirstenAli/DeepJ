package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/** Byte-faithful transport double; its only arithmetic is the existing NEG opcode. */
final class GpuStorageRuntime implements GpuRuntime {
    final Map<Integer, Object> values = new HashMap<>();
    final Map<Integer, DType> types = new HashMap<>();
    int downloads;

    @Override public void allocBuffers(int[] ids, int[] sizes, int count) {
        DType[] dtypes = new DType[count];
        Arrays.fill(dtypes, DType.FLOAT32);
        allocTypedBuffers(ids, sizes, dtypes, count);
    }

    @Override public void allocTypedBuffers(int[] ids, int[] sizes, DType[] dtypes, int count) {
        GpuAllocations.requireValid(ids, sizes, dtypes, count);
        for (int index = 0; index < count; index++) {
            values.put(ids[index], dtypes[index] == DType.FLOAT32 ? new float[sizes[index]] : new short[sizes[index]]);
            types.put(ids[index], dtypes[index]);
        }
    }

    @Override public void uploadBuffer(int id, float[] data) { copy(id, data, true); }
    @Override public void uploadBuffer16(int id, short[] data) { copy(id, data, true); }
    @Override public void downloadBuffer(int id, float[] data) {
        copy(id, data, false);
        downloads++;
    }
    @Override public void downloadBuffer16(int id, short[] data) {
        copy(id, data, false);
        downloads++;
    }

    private void copy(int id, Object data, boolean upload) {
        Object stored = values.get(id);
        int count = java.lang.reflect.Array.getLength(stored);
        if (upload) System.arraycopy(data, 0, stored, 0, count);
        else System.arraycopy(stored, 0, data, 0, count);
    }

    @Override public void releaseBuffers(int[] ids, int count) {
        for (int index = 0; index < count; index++) {
            values.remove(ids[index]);
            types.remove(ids[index]);
        }
    }

    @Override public void flushOps(int[] commands, int length) {
        for (int position = 0; position < length; position += 4) {
            if (commands[position] != ComputeGraph.OP_NEG) throw new AssertionError("Unexpected opcode");
            float[] input = (float[]) values.get(commands[position + 1]);
            float[] output = (float[]) values.get(commands[position + 2]);
            for (int index = 0; index < commands[position + 3]; index++) {
                output[index] = -input[index];
            }
        }
    }
}
