package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.ArrayList;
import java.util.List;

final class GradientBucket {
    private record Piece(Tensor target, int offset) {}
    private final float[] values;
    private final List<Piece> pieces = new ArrayList<>();
    private int size;

    GradientBucket(int capacity) { values = new float[capacity]; }
    int remaining() { return values.length - size; }
    boolean empty() { return size == 0; }

    void add(Tensor piece, float[] source, int offset) {
        System.arraycopy(source, offset, values, size, piece.numel());
        pieces.add(new Piece(piece, size));
        size += piece.numel();
    }

    void reduce(ProcessGroup group, String key, float scale) {
        float[] payload = size == values.length ? values : java.util.Arrays.copyOf(values, size);
        group.sum(key, payload);
        normalize(group, payload, scale);
        try (var scope = TensorScope.open()) {
            for (Piece piece : pieces) { restore(piece, payload); }
        }
        pieces.clear();
        size = 0;
    }

    private static void normalize(ProcessGroup group, float[] values, float scale) {
        for (int index = 0; index < values.length; index++) {
            values[index] *= scale;
            if (!Float.isFinite(values[index])) RankAgreement.fail(group, "Non-finite synchronized gradient");
        }
    }

    private static void restore(Piece piece, float[] values) {
        float[] source = java.util.Arrays.copyOfRange(values, piece.offset(), piece.offset() + piece.target().numel());
        piece.target().copyFrom(Tensor.from(source));
    }
}
