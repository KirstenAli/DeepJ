package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.nn.TensorJoin;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

final class SpatialOperators {
    private SpatialOperators() {}

    static void register(OnnxRegistry registry) {
        registry.add(
                "Conv",
                1,
                "auto_pad dilations group kernel_shape pads strides",
                context -> context.one(convolution(context)));
        registerPooling(registry);
        registry.add(
                "Pad", 2, "mode pads value", context -> List.of(PaddingOperators.pad(context)));
        registry.add(
                "ConvTranspose",
                1,
                "auto_pad dilations group kernel_shape output_padding pads strides",
                context -> context.one(TransposeConvolution.apply(context)));
        ResizeOperators.register(registry);
    }

    private static Tensor convolution(OnnxContext context) {
        Tensor input = context.tensor(0), weight = context.tensor(1);
        ValueOps.sameType(context.input(0), context.input(1));
        int[] kernel = Arrays.copyOfRange(weight.shape(), 2, weight.rank());
        if (!Arrays.equals(kernel, context.integers("kernel_shape", kernel)))
            throw new IllegalArgumentException("Conv kernel shape differs from weights");
        SpatialSpec spec =
                SpatialSpec.read(
                        context, Arrays.copyOfRange(input.shape(), 2, input.rank()), kernel);
        return convolve(
                input,
                weight,
                context.hasInput(2) ? context.tensor(2) : null,
                context.integer("group", 1),
                spec);
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
        Tensor output =
                groups == 1
                        ? results.get(0)
                        : TensorJoin.concatenate(1, results.toArray(Tensor[]::new));
        if (bias != null) output = output.add(bias.reshape(1, outputs, 1));
        return output.reshape(ValueOps.join(new int[] {input.size(0), outputs}, spec.output()));
    }

    private static Tensor group(
            Tensor patches, Tensor weight, int group, int groups, SpatialSpec spec) {
        int channels = patches.size(1) / groups, outputs = weight.size(0) / groups;
        Tensor input =
                patches.slice(1, group * channels, (group + 1) * channels)
                        .permute(0, 2, 1, 3)
                        .reshape(patches.size(0), spec.positions(), channels * spec.area());
        Tensor kernel =
                weight.slice(0, group * outputs, (group + 1) * outputs)
                        .reshape(outputs, channels * spec.area());
        return input.matmul(kernel.transpose()).transpose(1, 2);
    }

    private static Tensor patches(Tensor input, SpatialSpec spec, float fill) {
        int size = ValueOps.product(input.shape(), 2, input.rank());
        Tensor padding =
                Tensor.from(new float[] {fill}, input.dtype(), 1, 1, 1)
                        .expand(input.size(0), input.size(1), 1);
        Tensor source =
                TensorJoin.concatenate(
                        2, padding, input.reshape(input.size(0), input.size(1), size));
        return source.indexSelect(2, spec.indices())
                .reshape(input.size(0), input.size(1), spec.positions(), spec.area());
    }

    private static List<OnnxValue> pool(OnnxContext context) {
        Tensor input = context.tensor(0);
        boolean maximum = context.operator().equals("MaxPool");
        if (context.integer("storage_order", 0) != 0)
            throw new IllegalArgumentException("MaxPool storage_order must be zero");
        SpatialSpec spec =
                SpatialSpec.read(
                        context,
                        Arrays.copyOfRange(input.shape(), 2, input.rank()),
                        context.integers("kernel_shape"));
        Tensor windows = patches(input, spec, maximum ? Float.NEGATIVE_INFINITY : 0);
        Tensor output = maximum ? windows.maximum(-1, false) : averagePool(context, windows, spec);
        int[] shape = ValueOps.join(new int[] {input.size(0), input.size(1)}, spec.output());
        if (context.outputCount() == 1) return context.one(output.reshape(shape).to(input.dtype()));
        return List.of(
                OnnxValue.of(output.reshape(shape).to(input.dtype())),
                poolIndices(input, windows, spec, shape));
    }

    private static OnnxValue poolIndices(
            Tensor input, Tensor windows, SpatialSpec spec, int[] shape) {
        float[] values = windows.toFloatArray();
        int[] positions = spec.indices();
        long[] indices = new long[OnnxValue.elements(shape)];
        int spatial = ValueOps.product(input.shape(), 2, input.rank());
        for (int i = 0; i < indices.length; i++) {
            indices[i] = poolIndex(values, positions, i, spatial, spec);
        }
        return OnnxValue.integers(indices, shape);
    }

    private static long poolIndex(
            float[] values, int[] positions, int index, int spatial, SpatialSpec spec) {
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

    private static void registerPooling(OnnxRegistry registry) {
        registry.add(
                "MaxPool",
                8,
                "auto_pad ceil_mode dilations kernel_shape pads storage_order strides",
                SpatialOperators::pool);
        registry.add(
                "AveragePool",
                7,
                "auto_pad ceil_mode count_include_pad dilations kernel_shape pads strides",
                SpatialOperators::pool);
        registry.add(
                "GlobalAveragePool",
                1,
                "",
                context -> context.one(global(context.tensor(0), false)));
        registry.add(
                "GlobalMaxPool", 1, "", context -> context.one(global(context.tensor(0), true)));
    }

    private static Tensor averagePool(OnnxContext context, Tensor windows, SpatialSpec spec) {
        float[] counts = spec.counts(context.integer("count_include_pad", 0) != 0);
        return windows.sum(-1, false).divide(Tensor.from(counts, 1, 1, spec.positions()));
    }
}
