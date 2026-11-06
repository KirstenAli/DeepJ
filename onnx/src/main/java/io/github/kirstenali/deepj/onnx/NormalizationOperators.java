package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

final class NormalizationOperators {
    private NormalizationOperators() {}

    static void register(OnnxRegistry registry) {
        registry.add(
                "LayerNormalization", 17, "axis epsilon stash_type", NormalizationOperators::layer);
        registry.add(
                "InstanceNormalization", 6, "epsilon", context -> context.one(instance(context)));
        registry.add(
                "BatchNormalization",
                9,
                "epsilon momentum training_mode",
                NormalizationOperators::batch);
        registry.add(
                "GroupNormalization",
                18,
                "epsilon num_groups",
                context -> context.one(group(context)));
        registry.add("Dropout", 12, "seed", NormalizationOperators::dropout);
    }

    private static List<OnnxValue> layer(OnnxContext context) {
        if (context.integer("stash_type", 1) != 1)
            throw new IllegalArgumentException("LayerNormalization stash_type must be FLOAT");
        Tensor input = context.tensor(0), value = input.to(DType.FLOAT32);
        int axis = ValueOps.axis(context.integer("axis", -1), input.rank());
        int[] axes = IntStream.range(axis, input.rank()).toArray();
        Tensor mean = keepMean(value, axes), centered = value.subtract(mean);
        Tensor inverse =
                keepMean(centered.multiply(centered), axes)
                        .addScalar(context.floating("epsilon", 1e-5f))
                        .sqrt()
                        .pow(-1);
        Tensor output = centered.multiply(inverse).multiply(context.tensor(1).to(DType.FLOAT32));
        if (context.hasInput(2)) output = output.add(context.tensor(2).to(DType.FLOAT32));
        return List.of(
                OnnxValue.of(output.to(input.dtype())), OnnxValue.of(mean), OnnxValue.of(inverse));
    }

    private static Tensor keepMean(Tensor input, int[] axes) {
        Tensor value = input;
        for (int i = axes.length - 1; i >= 0; i--) {
            value = value.mean(axes[i], true);
        }
        return value;
    }

    private static Tensor instance(OnnxContext context) {
        Tensor value = context.tensor(0).to(DType.FLOAT32);
        if (value.rank() < 3)
            throw new IllegalArgumentException("InstanceNormalization needs spatial axes");
        int[] axes = IntStream.range(2, value.rank()).toArray(), shape = channelShape(value);
        Tensor output = standardize(value, axes, context.floating("epsilon", 1e-5f));
        return output.multiply(context.tensor(1).reshape(shape))
                .add(context.tensor(2).reshape(shape))
                .to(context.tensor(0).dtype());
    }

    private static int[] channelShape(Tensor input) {
        int[] shape = new int[input.rank()];
        Arrays.fill(shape, 1);
        shape[1] = input.size(1);
        return shape;
    }

    private static List<OnnxValue> batch(OnnxContext context) {
        Tensor value = context.tensor(0).to(DType.FLOAT32);
        boolean training =
                context.opset() < 14
                        ? context.outputCount() > 1
                        : context.integer("training_mode", 0) != 0;
        int[] axes = IntStream.range(0, value.rank()).filter(i -> i != 1).toArray();
        Tensor mean = training ? value.meanAxes(axes) : context.tensor(3);
        Tensor variance = training ? value.varianceAxes(axes, false, 0) : context.tensor(4);
        Tensor output = batchOutput(context, value, mean, variance);
        return training
                ? batchStatistics(context, output, mean, variance)
                : List.of(OnnxValue.of(output));
    }

    private static Tensor group(OnnxContext context) {
        Tensor input = context.tensor(0), value = input.to(DType.FLOAT32);
        int groups = context.integer("num_groups", 0);
        if (groups < 1 || input.size(1) % groups != 0) {
            throw new IllegalArgumentException("Invalid GroupNormalization groups");
        }
        int[] shape = groupShape(input, groups), axes = IntStream.range(2, shape.length).toArray();
        Tensor normalized =
                standardize(value.reshape(shape), axes, context.floating("epsilon", 1e-5f));
        if (context.opset() < 21)
            return groupAffine(context, normalized).reshape(input.shape()).to(input.dtype());
        return normalized
                .reshape(input.shape())
                .multiply(context.tensor(1).reshape(channelShape(input)))
                .add(context.tensor(2).reshape(channelShape(input)))
                .to(input.dtype());
    }

    private static Tensor groupAffine(OnnxContext context, Tensor input) {
        int[] shape = new int[input.rank()];
        Arrays.fill(shape, 1);
        shape[1] = input.size(1);
        return input.multiply(context.tensor(1).reshape(shape))
                .add(context.tensor(2).reshape(shape));
    }

    private static List<OnnxValue> dropout(OnnxContext context) {
        boolean training = context.hasInput(2) && context.input(2).scalar() != 0;
        float ratio = context.hasInput(1) ? context.tensor(1).item() : 0.5f;
        if (ratio < 0 || ratio >= 1 || !Float.isFinite(ratio))
            throw new IllegalArgumentException("Invalid Dropout ratio");
        OnnxValue condition = dropoutMask(context, training, ratio);
        Tensor output =
                training
                        ? BooleanOperators.where(
                                        condition,
                                        context.tensor(0),
                                        Tensor.backend()
                                                .zeros(
                                                        context.tensor(0).dtype(),
                                                        context.tensor(0).shape()))
                                .divideScalar(1 - ratio)
                        : context.tensor(0);
        return List.of(OnnxValue.of(output), condition);
    }

    private static OnnxValue dropoutMask(OnnxContext context, boolean training, float ratio) {
        boolean[] mask = new boolean[context.tensor(0).numel()];
        var random =
                context.session.random(
                        context.node, context.longAttribute("seed", System.nanoTime()));
        for (int i = 0; i < mask.length; i++) {
            mask[i] = !training || random.nextFloat() >= ratio;
        }
        return OnnxValue.booleans(mask, context.tensor(0).shape());
    }

    private static Tensor batchOutput(
            OnnxContext context, Tensor value, Tensor mean, Tensor variance) {
        int[] shape = channelShape(value);
        Tensor centered = value.subtract(mean.reshape(shape));
        Tensor denominator =
                variance.reshape(shape).addScalar(context.floating("epsilon", 1e-5f)).sqrt();
        return centered.divide(denominator)
                .multiply(context.tensor(1).reshape(shape))
                .add(context.tensor(2).reshape(shape))
                .to(context.tensor(0).dtype());
    }

    private static List<OnnxValue> batchStatistics(
            OnnxContext context, Tensor output, Tensor mean, Tensor variance) {
        float momentum = context.floating("momentum", 0.9f);
        Tensor runningMean =
                context.tensor(3).multiplyScalar(momentum).add(mean.multiplyScalar(1 - momentum));
        Tensor runningVar =
                context.tensor(4)
                        .multiplyScalar(momentum)
                        .add(variance.multiplyScalar(1 - momentum));
        return List.of(
                OnnxValue.of(output),
                OnnxValue.of(runningMean),
                OnnxValue.of(runningVar),
                OnnxValue.of(mean),
                OnnxValue.of(variance));
    }

    private static Tensor standardize(Tensor value, int[] axes, float epsilon) {
        Tensor centered = value.subtract(keepMean(value, axes));
        return centered.divide(
                keepMean(centered.multiply(centered), axes).addScalar(epsilon).sqrt());
    }

    private static int[] groupShape(Tensor input, int groups) {
        return ValueOps.join(
                new int[] {input.size(0), groups, input.size(1) / groups},
                Arrays.copyOfRange(input.shape(), 2, input.rank()));
    }
}
