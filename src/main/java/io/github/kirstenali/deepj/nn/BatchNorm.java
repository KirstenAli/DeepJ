package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/** Channel-axis (axis 1) normalization, using backend reductions and persistent FLOAT32 running statistics. */
public final class BatchNorm extends Module {
    private final int channels;
    private final float epsilon;
    private final float momentum;
    private final Parameter gain;
    private final Parameter bias;
    private final Tensor runningMean;
    private final Tensor runningVariance;

    public BatchNorm(int channels) { this(channels, 1e-5f, 0.1f, ParameterFactory.float32()); }

    public BatchNorm(int channels, float epsilon, float momentum, ParameterFactory parameters) {
        this.channels = NeuralChecks.positive(channels, "channels");
        if (!Float.isFinite(epsilon) || epsilon <= 0) throw new IllegalArgumentException("Epsilon must be positive");
        if (!Float.isFinite(momentum) || momentum < 0 || momentum > 1) {
            throw new IllegalArgumentException("Momentum must be in [0, 1]");
        }
        this.epsilon = epsilon;
        this.momentum = momentum;
        gain = parameters.ones("gain", 1, channels);
        bias = parameters.zeros("bias", 1, channels);
        runningMean = Tensor.backend().zeros(DType.FLOAT32, 1, channels).retainDeviceBuffer();
        runningVariance = Tensor.backend().zeros(DType.FLOAT32, 1, channels).addScalar(1).retainDeviceBuffer();
    }

    @Override public List<Parameter> parameters() { return List.of(gain, bias); }
    @Override public List<Tensor> buffers() { return List.of(runningMean, runningVariance); }
    public Tensor runningMean() { return runningMean; }
    public Tensor runningVariance() { return runningVariance; }

    @Override public Tensor forward(Tensor input) {
        int[] shape = broadcastShape(input);
        try (var scope = TensorScope.open()) {
            Tensor values = input.to(DType.FLOAT32);
            Tensor[] statistics = isTraining() ? statistics(values) : new Tensor[]{runningMean, runningVariance};
            Tensor centered = values.subtract(statistics[0].reshape(shape));
            Tensor normalized = centered.divide(statistics[1].reshape(shape).addScalar(epsilon).sqrt());
            return scope.escape(normalized.multiply(gain.value.reshape(shape)).add(bias.value.reshape(shape)));
        }
    }

    private int[] broadcastShape(Tensor input) {
        if (input == null || input.rank() < 2 || input.numel() == 0 || input.size(1) != channels) {
            throw new IllegalArgumentException("BatchNorm requires nonempty [batch, channels, ...] input");
        }
        int[] shape = new int[input.rank()];
        Arrays.fill(shape, 1);
        shape[1] = channels;
        return shape;
    }

    private Tensor[] statistics(Tensor input) {
        int count = input.numel() / channels;
        if (count < 2) throw new IllegalArgumentException("Training BatchNorm needs two values per channel");
        int[] axes = IntStream.range(0, input.rank()).filter(axis -> axis != 1).toArray();
        Tensor mean = input.meanAxes(axes).reshape(1, channels);
        Tensor variance = input.varianceAxes(axes, false, 0).reshape(1, channels);
        update(mean, variance, count);
        return new Tensor[]{mean, variance};
    }

    private void update(Tensor mean, Tensor variance, int count) {
        try (var ignored = GradMode.noGrad()) {
            runningMean.copyFrom(runningMean.multiplyScalar(1 - momentum).add(mean.multiplyScalar(momentum)));
            float correction = (float) count / (count - 1);
            runningVariance.copyFrom(runningVariance.multiplyScalar(1 - momentum)
                    .add(variance.multiplyScalar(momentum * correction)));
        }
    }
}
