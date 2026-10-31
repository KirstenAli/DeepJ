package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/** Byte-faithful transport/copy double; its only arithmetic is the existing NEG opcode. */
final class GpuStorageRuntime implements GpuRuntime {
    final Map<Integer, Object> values = new HashMap<>();
    final Map<Integer, DType> types = new HashMap<>();
    int downloads;
    long[] lastLayouts;

    @Override public boolean supportsLayouts() { return true; }

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
        int position = 0;
        while (position < length) {
            if (commands[position] == ComputeGraph.OP_STORAGE_COPY) {
                System.arraycopy(values.get(commands[position + 1]), commands[position + 3],
                        values.get(commands[position + 2]), commands[position + 4], commands[position + 5]);
                position += 6;
            } else {
                negate(commands, position);
                position += 4;
            }
        }
    }

    @Override public void flushOps(int[] commands, int length, long[] layouts) {
        lastLayouts = layouts.clone();
        int position = 0;
        while (position < length) {
            int stride = GraphExecution.getOpStride(commands[position]);
            if (commands[position] == ComputeGraph.OP_LAYOUT_TRANSFER) transfer(commands, position, layouts);
            else flushOps(Arrays.copyOfRange(commands, position, position + stride), stride);
            position += stride;
        }
    }

    private void transfer(int[] commands, int position, long[] layouts) {
        int input = commands[position + 1], output = commands[position + 2], start = commands[position + 3];
        int count = Math.toIntExact(layouts[start + 1]);
        for (int index = 0; index < count; index++) {
            int address = logicalAddress(layouts, start, index);
            if (types.get(input) == types.get(output)) {
                System.arraycopy(values.get(input), address, values.get(output), index, 1);
            } else assignCast(input, output, address, index);
        }
    }

    private void assignCast(int input, int output, int address, int index) {
        Object source = values.get(input), target = values.get(output);
        float value = source instanceof float[] floats ? floats[address]
                : types.get(input).decode16(((short[]) source)[address]);
        if (target instanceof float[] floats) floats[index] = value;
        else ((short[]) target)[index] = types.get(output).encode16(value);
    }

    private static int logicalAddress(long[] words, int start, int index) {
        int rank = Math.toIntExact(words[start]);
        long address = words[start + 3];
        for (int axis = rank - 1; axis >= 0; axis--) {
            int dimension = Math.toIntExact(words[start + 4 + axis]);
            address += index % dimension * words[start + 4 + rank + axis];
            index /= dimension;
        }
        return Math.toIntExact(address);
    }

    private void negate(int[] commands, int position) {
        if (commands[position] != ComputeGraph.OP_NEG) throw new AssertionError("Unexpected opcode");
        float[] input = (float[]) values.get(commands[position + 1]);
        float[] output = (float[]) values.get(commands[position + 2]);
        for (int index = 0; index < commands[position + 3]; index++) {
            output[index] = -input[index];
        }
    }
}
