package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Immutable masks built from metadata, without inspecting model tensors. True means a valid token. */
record EncoderMask(Tensor keys, Tensor queries, Tensor counts) {
    static EncoderMask create(Tensor input, boolean[][] valid) {
        NeuralChecks.rank(input, 3);
        if (valid == null) return null;
        if (valid.length != input.size(0)) throw new IllegalArgumentException("Mask batch size differs");
        int batch = input.size(0), time = input.size(1);
        float[] keys = new float[Math.multiplyExact(batch, time)], queries = new float[keys.length];
        float[] counts = new float[batch];
        for (int sample = 0; sample < batch; sample++) { fill(valid[sample], sample, time, keys, queries, counts); }
        return new EncoderMask(Tensor.from(keys, batch, 1, 1, time), Tensor.from(queries, batch, time, 1),
                Tensor.from(counts, batch, 1));
    }

    private static void fill(boolean[] valid, int sample, int time, float[] keys, float[] queries, float[] counts) {
        if (valid == null || valid.length != time) throw new IllegalArgumentException("Mask time dimension differs");
        for (int position = 0; position < time; position++) {
            int index = sample * time + position;
            keys[index] = valid[position] ? 0 : Float.NEGATIVE_INFINITY;
            queries[index] = valid[position] ? 1 : 0;
            if (valid[position]) counts[sample]++;
        }
        if (counts[sample] == 0) throw new IllegalArgumentException("Each sample needs at least one valid token");
    }

    Tensor apply(Tensor output) { return output.multiply(queries); }
}
