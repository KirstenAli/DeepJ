package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.LayoutOps;

/** Batch and head axes are distinct; packed matrices are temporary native-kernel views only. */
public final class HeadLayout {
    private HeadLayout() {}

    public static Tensor split(Tensor input, int heads) {
        if (input.rank() != 3 || heads < 1 || input.size(2) < 1 || input.size(2) % heads != 0) {
            throw new IllegalArgumentException("Head splitting requires[batch,time,heads*width]");
        }
        int batch = input.size(0), time = input.size(1), width = input.size(2) / heads;
        if (batch == 1 && fullFloat32(input)) {
            return LayoutOps.splitHeads(input.reshape(time, input.size(2)), heads).reshape(1, heads, time, width);
        }
        return input.reshape(batch, time, heads, width).permute(0, 2, 1, 3).contiguous();
    }

    public static Tensor merge(Tensor input, int heads) {
        require(input, heads);
        int batch = input.size(0), time = input.size(2), width = Math.multiplyExact(heads, input.size(3));
        if (batch == 1 && fullFloat32(input)) {
            return LayoutOps.mergeHeads(pack(input), heads).reshape(1, time, width);
        }
        return input.permute(0, 2, 1, 3).reshape(batch, time, width);
    }

    public static Tensor pack(Tensor input) {
        require(input, input.rank() == 4 ? input.size(1) : 0);
        return input.reshape(input.numel() / input.size(3), input.size(3));
    }

    public static void require(Tensor input, int heads) {
        if (input.rank() != 4 || heads < 1 || input.size(1) != heads || input.size(3) < 1) {
            throw new IllegalArgumentException("Expected[batch,heads,time,headWidth]activations");
        }
    }

    public static int totalHeads(Tensor input, int heads) {
        require(input, heads);
        return Math.multiplyExact(input.size(0), heads);
    }

    private static boolean fullFloat32(Tensor input) {
        return input.numel() > 0 && input.dtype() == DType.FLOAT32 && input.isContiguous()
                && input.storageOffset() == 0 && input.storageBytes() == (long) input.numel() * Float.BYTES;
    }
}
