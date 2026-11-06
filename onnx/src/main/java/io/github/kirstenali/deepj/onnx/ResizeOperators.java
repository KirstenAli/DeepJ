package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Arrays;
import java.util.stream.IntStream;

final class ResizeOperators {
    private ResizeOperators() {}

    static void register(OnnxRegistry registry) {
        registry.add(
                "Resize",
                10,
                "antialias axes coordinate_transformation_mode cubic_coeff_a exclude_outside"
                        + " extrapolation_value keep_aspect_ratio_policy mode nearest_mode",
                context -> context.one(resize(context)));
        registry.add("Upsample", 7, "mode scales", context -> context.one(resize(context)));
    }

    private static Tensor resize(OnnxContext context) {
        if (context.integer("antialias", 0) != 0
                || !context.string("keep_aspect_ratio_policy", "stretch").equals("stretch")) {
            throw new IllegalArgumentException(
                    "Resize requires antialias=0 and stretch aspect ratio");
        }
        Tensor input = context.tensor(0);
        int[] axes = context.integers("axes", IntStream.range(0, input.rank()).toArray());
        axes = Arrays.stream(axes).map(i -> ValueOps.axis(i, input.rank())).toArray();
        if (Arrays.stream(axes).distinct().count() != axes.length)
            throw new IllegalArgumentException("Duplicate Resize axis");
        double[] scales = scales(context, input, axes);
        int[] sizes = sizes(context, input, axes, scales);
        Tensor result = input;
        for (int i = 0; i < axes.length; i++) {
            result = axis(result, axes[i], sizes[i], scales[i], context);
        }
        return result;
    }

    private static double[] scales(OnnxContext context, Tensor input, int[] axes) {
        int position = legacy(context) ? 1 : 2;
        boolean hasScales = nonempty(context, position), hasSizes = nonempty(context, 3);
        if (hasScales && hasSizes)
            throw new IllegalArgumentException("Resize accepts either scales or sizes");
        if (hasScales) {
            float[] values = context.tensor(position).toFloatArray();
            if (values.length != axes.length)
                throw new IllegalArgumentException("Resize scales rank differs");
            return IntStream.range(0, values.length).mapToDouble(i -> values[i]).toArray();
        }
        if (hasSizes) return sizeScales(context.input(3).ints(), input, axes);
        return attributeScales(context, axes.length);
    }

    private static int[] sizes(OnnxContext context, Tensor input, int[] axes, double[] scales) {
        if (context.hasInput(3) && context.input(3).size() != 0) return context.input(3).ints();
        return IntStream.range(0, axes.length)
                .map(
                        i -> {
                            if (!Double.isFinite(scales[i]) || scales[i] <= 0)
                                throw new IllegalArgumentException(
                                        "Resize scale must be finite and positive");
                            return Math.toIntExact(
                                    (long) Math.floor(input.size(axes[i]) * scales[i]));
                        })
                .toArray();
    }

    private static Tensor axis(
            Tensor input, int axis, int outputSize, double scale, OnnxContext context) {
        String mode = context.string("mode", "nearest");
        int[] lower = new int[outputSize], upper = new int[outputSize];
        float[] weights = new float[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double position = position(i, input.size(axis), outputSize, scale, context);
            if (mode.equals("nearest")) lower[i] = nearest(position, input.size(axis), context);
            else {
                position = Math.max(0, Math.min(input.size(axis) - 1, position));
                lower[i] = (int) Math.floor(position);
                upper[i] = Math.min(input.size(axis) - 1, lower[i] + 1);
                weights[i] = (float) (position - lower[i]);
            }
        }
        if (mode.equals("nearest")) return input.indexSelect(axis, lower);
        if (!mode.equals("linear"))
            throw new IllegalArgumentException("Supported Resize modes: nearest, linear");
        return linear(input, axis, lower, upper, weights);
    }

    private static Tensor linear(
            Tensor input, int axis, int[] lower, int[] upper, float[] weights) {
        int[] shape = new int[input.rank()];
        Arrays.fill(shape, 1);
        shape[axis] = weights.length;
        Tensor fraction = Tensor.from(weights, input.dtype(), shape);
        return input.indexSelect(axis, lower)
                .multiply(fraction.neg().addScalar(1))
                .add(input.indexSelect(axis, upper).multiply(fraction));
    }

    private static double position(
            int coordinate, int input, int output, double scale, OnnxContext context) {
        String mode =
                context.string(
                        "coordinate_transformation_mode",
                        legacy(context) ? "asymmetric" : "half_pixel");
        return switch (mode) {
            case "asymmetric" -> coordinate / scale;
            case "half_pixel" -> (coordinate + 0.5) / scale - 0.5;
            case "pytorch_half_pixel" -> output > 1 ? (coordinate + 0.5) / scale - 0.5 : 0;
            default -> extendedPosition(mode, coordinate, input, output, scale);
        };
    }

    private static int nearest(double position, int size, OnnxContext context) {
        String mode =
                context.string(
                        "nearest_mode",
                        context.opset() == 10 || context.operator().equals("Upsample")
                                ? "floor"
                                : "round_prefer_floor");
        double index =
                switch (mode) {
                    case "floor" -> Math.floor(position);
                    case "ceil" -> Math.ceil(position);
                    case "round_prefer_floor" -> Math.ceil(position - 0.5);
                    case "round_prefer_ceil" -> Math.floor(position + 0.5);
                    default ->
                            throw new IllegalArgumentException("Unsupported nearest_mode " + mode);
                };
        return (int) Math.max(0, Math.min(size - 1, index));
    }

    private static double[] sizeScales(int[] sizes, Tensor input, int[] axes) {
        if (sizes.length != axes.length)
            throw new IllegalArgumentException("Resize sizes rank differs");
        return IntStream.range(0, axes.length)
                .mapToDouble(i -> (double) sizes[i] / input.size(axes[i]))
                .toArray();
    }

    private static boolean nonempty(OnnxContext context, int position) {
        return context.hasInput(position) && context.input(position).size() != 0;
    }

    private static boolean legacy(OnnxContext context) {
        return context.opset() == 10 || context.operator().equals("Upsample");
    }

    private static double extendedPosition(
            String mode, int coordinate, int input, int output, double scale) {
        return switch (mode) {
            case "align_corners" -> output > 1 ? coordinate * (input - 1.0) / (output - 1) : 0;
            case "tf_half_pixel_for_nn" -> (coordinate + 0.5) / scale;
            case "half_pixel_symmetric" ->
                    input * 0.5 * (1 - output / (input * scale)) + (coordinate + 0.5) / scale - 0.5;
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported Resize coordinate mode " + mode);
        };
    }

    private static double[] attributeScales(OnnxContext context, int rank) {
        var values = context.attributes.get("scales");
        if (values == null || values.getFloatsCount() != rank) {
            throw new IllegalArgumentException("Resize needs scales or sizes");
        }
        return values.getFloatsList().stream().mapToDouble(Float::doubleValue).toArray();
    }
}
