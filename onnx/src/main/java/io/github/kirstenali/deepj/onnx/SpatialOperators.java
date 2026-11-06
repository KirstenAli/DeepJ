package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.nn.TensorJoin;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

final class SpatialOperators {
    private SpatialOperators() {}

    static void register(OnnxRegistry r) {
        r.add("Conv", 1, "auto_pad dilations group kernel_shape pads strides", c -> c.one(convolution(c)));
        r.add("MaxPool", 8, "auto_pad ceil_mode dilations kernel_shape pads storage_order strides",
                SpatialOperators::pool);
        r.add("AveragePool", 7, "auto_pad ceil_mode count_include_pad dilations kernel_shape pads strides",
                SpatialOperators::pool);
        r.add("GlobalAveragePool", 1, "", c -> c.one(global(c.tensor(0), false)));
        r.add("GlobalMaxPool", 1, "", c -> c.one(global(c.tensor(0), true)));
        r.add("Pad", 2, "mode pads value", c -> List.of(PaddingOperators.pad(c)));
        r.add("ConvTranspose", 1, "auto_pad dilations group kernel_shape output_padding pads strides",
                c -> c.one(TransposeConvolution.apply(c)));
        ResizeOperators.register(r);
    }

    private static Tensor convolution(OnnxContext c) {
        Tensor input = c.tensor(0), weight = c.tensor(1);
        ValueOps.sameType(c.input(0), c.input(1));
        int[] kernel = Arrays.copyOfRange(weight.shape(), 2, weight.rank());
        if (!Arrays.equals(kernel, c.integers("kernel_shape", kernel)))
            throw new IllegalArgumentException("Conv kernel shape differs from weights");
        SpatialSpec spec = SpatialSpec.read(c, Arrays.copyOfRange(input.shape(), 2, input.rank()), kernel);
        return convolve(input, weight, c.hasInput(2) ? c.tensor(2) : null, c.integer("group", 1), spec);
    }

    static Tensor convolve(Tensor input, Tensor weight, Tensor bias, int groups, SpatialSpec spec) {
        int outputs = weight.size(0), channels = input.size(1);
        if (groups < 1
                || channels % groups != 0
                || outputs % groups != 0
                || weight.size(1) != channels / groups) {
            throw new IllegalArgumentException("Conv groups or channel dimensions differ");
        }
        Tensor patches = patches(input, spec, 0);
        var results = new ArrayList<Tensor>();
        for (int group = 0; group < groups; group++) {
            results.add(group(patches, weight, group, groups, spec));
        }
        Tensor output = groups == 1 ? results.get(0) : TensorJoin.concatenate(1, results.toArray(Tensor[]::new));
        if (bias != null) output = output.add(bias.reshape(1, outputs, 1));
        return output.reshape(ValueOps.join(new int[] {input.size(0), outputs}, spec.output()));
    }

    private static Tensor group(Tensor patches, Tensor weight, int group, int groups, SpatialSpec spec) {
        int channels = patches.size(1) / groups, outputs = weight.size(0) / groups;
        Tensor input = patches.slice(1, group * channels, (group + 1) * channels).permute(0, 2, 1, 3)
                        .reshape(patches.size(0), spec.positions(), channels * spec.area());
        Tensor kernel = weight.slice(0, group * outputs, (group + 1) * outputs)
                        .reshape(outputs, channels * spec.area());
        return input.matmul(kernel.transpose()).transpose(1, 2);
    }

    private static Tensor patches(Tensor input, SpatialSpec spec, float fill) {
        int size = ValueOps.product(input.shape(), 2, input.rank());
        Tensor padding = Tensor.from(new float[] {fill}, input.dtype(), 1, 1, 1)
                        .expand(input.size(0), input.size(1), 1);
        Tensor source = TensorJoin.concatenate(2, padding, input.reshape(input.size(0), input.size(1), size));
        return source.indexSelect(2, spec.indices())
                .reshape(input.size(0), input.size(1), spec.positions(), spec.area());
    }

    private static List<OnnxValue> pool(OnnxContext c) {
        Tensor input = c.tensor(0);
        boolean maximum = c.operator().equals("MaxPool");
        if (c.integer("storage_order", 0) != 0)
            throw new IllegalArgumentException("MaxPool storage_order must be zero");
        SpatialSpec spec = SpatialSpec.read(c, Arrays.copyOfRange(input.shape(), 2, input.rank()),
                        c.integers("kernel_shape"));
        Tensor windows = patches(input, spec, maximum ? Float.NEGATIVE_INFINITY : 0);
        Tensor output = maximum ? windows.maximum(-1, false) : windows.sum(-1, false).divide(Tensor.from(
                                                spec.counts(c.integer("count_include_pad", 0) != 0), 1, 1,
                                                spec.positions()));
        int[] shape = ValueOps.join(new int[] {input.size(0), input.size(1)}, spec.output());
        if (c.outputCount() == 1) return c.one(output.reshape(shape).to(input.dtype()));
        return List.of(OnnxValue.of(output.reshape(shape).to(input.dtype())), poolIndices(input, windows, spec, shape));
    }

    private static OnnxValue poolIndices(Tensor input, Tensor windows, SpatialSpec spec, int[] shape) {
        float[] values = windows.toFloatArray();
        int[] positions = spec.indices();
        long[] indices = new long[OnnxValue.elements(shape)];
        int spatial = ValueOps.product(input.shape(), 2, input.rank());
        for (int i = 0; i < indices.length; i++) {
            indices[i] = poolIndex(values, positions, i, spatial, spec);
        }
        return OnnxValue.integers(indices, shape);
    }

    private static long poolIndex(float[] values, int[] positions, int index, int spatial, SpatialSpec spec) {
        int best = -1, offset = index * spec.area();
        for (int tap = 0; tap < spec.area(); tap++) {
            int source = index % spec.positions() * spec.area() + tap;
            if (positions[source] != 0
                    && (best < 0 || values[offset + tap] > values[offset + best])) best = tap;
        }
        if (best < 0) throw new IllegalArgumentException("MaxPool window contains only padding");
        return (long) (index / spec.positions()) * spatial
                + positions[index % spec.positions() * spec.area() + best]
                - 1;
    }

    private static Tensor global(Tensor input, boolean maximum) {
        if (input.rank() < 3)
            throw new IllegalArgumentException("Global pooling requires spatial dimensions");
        Tensor result = input;
        for (int axis = input.rank() - 1; axis >= 2; axis--) {
            result = maximum ? result.maximum(axis, true) : result.mean(axis, true);
        }
        return result;
    }
}
