package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.nn.TensorJoin;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.Arrays;
import java.util.stream.IntStream;

final class TransposeConvolution {
    private TransposeConvolution() {}

    static Tensor apply(OnnxContext c) {
        Tensor input = c.tensor(0), original = c.tensor(1);
        ValueOps.sameType(c.input(0), c.input(1));
        int rank = input.rank() - 2, groups = c.integer("group", 1);
        int[] kernel = Arrays.copyOfRange(original.shape(), 2, original.rank());
        if (!Arrays.equals(kernel, c.integers("kernel_shape", kernel))) {
            throw new IllegalArgumentException("ConvTranspose kernel shape differs from weights");
        }
        validateChannels(input, original, groups);
        int[] stride = c.integers("strides", SpatialSpec.ones(rank));
        if (stride.length != rank)
            throw new IllegalArgumentException("ConvTranspose stride rank differs");
        for (int i = 0; i < rank; i++) {
            input = upsample(input, i + 2, stride[i]);
        }
        var spec = geometry(c, input, kernel, stride);
        Tensor bias = c.hasInput(2) ? c.tensor(2) : null;
        return SpatialOperators.convolve(input, transposeWeight(original, groups, kernel), bias, groups, spec);
    }

    private static Tensor transposeWeight(Tensor input, int groups, int[] kernel) {
        int[] grouped = ValueOps.join(new int[] {groups, input.size(0) / groups, input.size(1)}, kernel);
        int[] permutation = IntStream.range(0, grouped.length).toArray();
        permutation[1] = 2;
        permutation[2] = 1;
        Tensor weight = input.reshape(grouped).permute(permutation).reshape(ValueOps.join(
                                        new int[] {groups * input.size(1), input.size(0) / groups}, kernel));
        for (int i = 0; i < kernel.length; i++) {
            int width = kernel[i];
            weight = weight.indexSelect(i + 2, IntStream.range(0, width).map(j -> width - j - 1).toArray());
        }
        return weight;
    }

    private static Tensor upsample(Tensor input, int axis, int stride) {
        if (stride < 1 || input.size(axis) < 1)
            throw new IllegalArgumentException("Invalid ConvTranspose stride or empty input");
        if (stride == 1) return input;
        Tensor[] pieces = new Tensor[stride];
        Arrays.fill(pieces, Tensor.backend().zeros(input.dtype(), input.shape()));
        pieces[0] = input;
        int[] shape = input.shape();
        shape[axis] = Math.multiplyExact(shape[axis], stride);
        return TensorJoin.stack(axis + 1, pieces).reshape(shape).slice(axis, 0, shape[axis] - stride + 1);
    }

    private static void setPadding(OnnxContext c, int i, int[] kernel, int[] stride, int[] dilation, int[] pads,
            int[] extra, int[] before, int[] after) {
        if (extra[i] < 0 || extra[i] >= Math.max(stride[i], dilation[i]))
            throw new IllegalArgumentException("Invalid output_padding");
        int full = SpatialSpec.effective(kernel[i], dilation[i]) - 1;
        String auto = c.string("auto_pad", "NOTSET");
        if (auto.equals("VALID")) {
            pads[i] = 0;
            pads[i + kernel.length] = 0;
        } else if (auto.equals("SAME_UPPER") || auto.equals("SAME_LOWER")) {
            int total = full + extra[i] + 1 - stride[i];
            pads[i] = auto.equals("SAME_LOWER") ? (total + 1) / 2 : total / 2;
            pads[i + kernel.length] = total - pads[i];
        } else if (!auto.equals("NOTSET")) throw new IllegalArgumentException("Invalid auto_pad");
        before[i] = full - pads[i];
        after[i] = full - pads[i + kernel.length] + extra[i];
    }

    private static void validateChannels(Tensor input, Tensor weight, int groups) {
        if (weight.size(0) != input.size(1) || groups < 1 || input.size(1) % groups != 0) {
            throw new IllegalArgumentException("ConvTranspose channels differ");
        }
    }

    private static SpatialSpec geometry(OnnxContext c, Tensor input, int[] kernel, int[] stride) {
        int rank = kernel.length;
        int[] dilation = c.integers("dilations", SpatialSpec.ones(rank));
        int[] pads = c.integers("pads", new int[2 * rank]), extra = c.integers("output_padding", new int[rank]);
        if (dilation.length != rank || pads.length != 2 * rank || extra.length != rank) {
            throw new IllegalArgumentException("ConvTranspose spatial attribute ranks differ");
        }
        int[] before = new int[rank], after = new int[rank];
        for (int i = 0; i < rank; i++) {
            setPadding(c, i, kernel, stride, dilation, pads, extra, before, after);
        }
        return SpatialSpec.create(Arrays.copyOfRange(input.shape(), 2, input.rank()), kernel, SpatialSpec.ones(rank),
                dilation, before, after, false);
    }
}
