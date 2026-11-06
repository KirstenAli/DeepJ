package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

final class NormalizationOperators {
    private NormalizationOperators() {}

    static void register(OnnxRegistry r) {
        r.add("LayerNormalization", 17, "axis epsilon stash_type", NormalizationOperators::layer);
        r.add("InstanceNormalization", 6, "epsilon", c -> c.one(instance(c)));
        r.add("BatchNormalization", 9, "epsilon momentum training_mode", NormalizationOperators::batch);
        r.add("GroupNormalization", 18, "epsilon num_groups", c -> c.one(group(c)));
        r.add("Dropout", 12, "seed", NormalizationOperators::dropout);
    }

    private static List<OnnxValue> layer(OnnxContext c) {
        if (c.integer("stash_type", 1) != 1)
            throw new IllegalArgumentException("LayerNormalization stash_type must be FLOAT");
        Tensor input = c.tensor(0), value = input.to(DType.FLOAT32);
        int axis = ValueOps.axis(c.integer("axis", -1), input.rank());
        int[] axes = IntStream.range(axis, input.rank()).toArray();
        Tensor mean = keepMean(value, axes), centered = value.subtract(mean);
        Tensor inverse = keepMean(centered.multiply(centered), axes).addScalar(c.floating("epsilon", 1e-5f)).sqrt()
                        .pow(-1);
        Tensor output = centered.multiply(inverse).multiply(c.tensor(1).to(DType.FLOAT32));
        if (c.hasInput(2)) output = output.add(c.tensor(2).to(DType.FLOAT32));
        return List.of(OnnxValue.of(output.to(input.dtype())), OnnxValue.of(mean), OnnxValue.of(inverse));
    }

    private static Tensor keepMean(Tensor input, int[] axes) {
        Tensor value = input;
        for (int i = axes.length - 1; i >= 0; i--) {
            value = value.mean(axes[i], true);
        }
        return value;
    }

    private static Tensor instance(OnnxContext c) {
        Tensor value = c.tensor(0).to(DType.FLOAT32);
        if (value.rank() < 3)
            throw new IllegalArgumentException("InstanceNormalization needs spatial axes");
        int[] axes = IntStream.range(2, value.rank()).toArray();
        Tensor centered = value.subtract(keepMean(value, axes));
        Tensor output = centered.divide(keepMean(centered.multiply(centered), axes)
                                .addScalar(c.floating("epsilon", 1e-5f)).sqrt());
        int[] shape = channelShape(value);
        return output.multiply(c.tensor(1).reshape(shape)).add(c.tensor(2).reshape(shape)).to(c.tensor(0).dtype());
    }

    private static int[] channelShape(Tensor input) {
        int[] shape = new int[input.rank()];
        Arrays.fill(shape, 1);
        shape[1] = input.size(1);
        return shape;
    }

    private static List<OnnxValue> batch(OnnxContext c) {
        Tensor value = c.tensor(0).to(DType.FLOAT32);
        boolean training = c.opset() < 14 ? c.outputCount() > 1 : c.integer("training_mode", 0) != 0;
        int[] axes = IntStream.range(0, value.rank()).filter(i -> i != 1).toArray(), shape = channelShape(value);
        Tensor mean = training ? value.meanAxes(axes) : c.tensor(3);
        Tensor variance = training ? value.varianceAxes(axes, false, 0) : c.tensor(4);
        Tensor output = value.subtract(mean.reshape(shape)).divide(variance.reshape(shape)
                                        .addScalar(c.floating("epsilon", 1e-5f)).sqrt())
                        .multiply(c.tensor(1).reshape(shape)).add(c.tensor(2).reshape(shape)).to(c.tensor(0).dtype());
        if (!training) return List.of(OnnxValue.of(output));
        float momentum = c.floating("momentum", 0.9f);
        Tensor runningMean = c.tensor(3).multiplyScalar(momentum).add(mean.multiplyScalar(1 - momentum));
        Tensor runningVar = c.tensor(4).multiplyScalar(momentum).add(variance.multiplyScalar(1 - momentum));
        return List.of(OnnxValue.of(output), OnnxValue.of(runningMean), OnnxValue.of(runningVar), OnnxValue.of(mean),
                OnnxValue.of(variance));
    }

    private static Tensor group(OnnxContext c) {
        Tensor input = c.tensor(0), value = input.to(DType.FLOAT32);
        int groups = c.integer("num_groups", 0);
        if (groups < 1 || input.size(1) % groups != 0)
            throw new IllegalArgumentException("Invalid GroupNormalization groups");
        int[] shape = ValueOps.join(new int[] {input.size(0), groups, input.size(1) / groups},
                        Arrays.copyOfRange(input.shape(), 2, input.rank()));
        Tensor grouped = value.reshape(shape);
        int[] axes = IntStream.range(2, shape.length).toArray();
        Tensor centered = grouped.subtract(keepMean(grouped, axes));
        Tensor normalized = centered.divide(keepMean(centered.multiply(centered), axes)
                                .addScalar(c.floating("epsilon", 1e-5f)).sqrt());
        if (c.opset() < 21)
            return groupAffine(c, normalized).reshape(input.shape()).to(input.dtype());
        return normalized.reshape(input.shape()).multiply(c.tensor(1).reshape(channelShape(input)))
                .add(c.tensor(2).reshape(channelShape(input))).to(input.dtype());
    }

    private static Tensor groupAffine(OnnxContext c, Tensor input) {
        int[] shape = new int[input.rank()];
        Arrays.fill(shape, 1);
        shape[1] = input.size(1);
        return input.multiply(c.tensor(1).reshape(shape)).add(c.tensor(2).reshape(shape));
    }

    private static List<OnnxValue> dropout(OnnxContext c) {
        boolean training = c.hasInput(2) && c.input(2).scalar() != 0;
        float ratio = c.hasInput(1) ? c.tensor(1).item() : 0.5f;
        if (ratio < 0 || ratio >= 1 || !Float.isFinite(ratio))
            throw new IllegalArgumentException("Invalid Dropout ratio");
        OnnxValue condition = dropoutMask(c, training, ratio);
        Tensor output = training ? BooleanOperators.where(condition, c.tensor(0), Tensor.backend()
                                                .zeros(c.tensor(0).dtype(), c.tensor(0).shape()))
                                .divideScalar(1 - ratio) : c.tensor(0);
        return List.of(OnnxValue.of(output), condition);
    }

    private static OnnxValue dropoutMask(OnnxContext c, boolean training, float ratio) {
        boolean[] mask = new boolean[c.tensor(0).numel()];
        var random = c.session.random(c.node, c.longAttribute("seed", System.nanoTime()));
        for (int i = 0; i < mask.length; i++) {
            mask[i] = !training || random.nextFloat() >= ratio;
        }
        return OnnxValue.booleans(mask, c.tensor(0).shape());
    }
}
