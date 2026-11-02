package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.List;

final class GradientReducer {
    private final ProcessGroup group;
    private final int capacity;

    GradientReducer(ProcessGroup group, int capacity) {
        this.group = group;
        this.capacity = capacity;
    }

    void average(List<Parameter> parameters, long totalWeight) {
        try (var ignored = GradMode.noGrad(); var scope = TensorScope.open()) {
            GradientBucket bucket = new GradientBucket(capacity);
            int index = 0;
            for (Parameter parameter : parameters) {
                if (!parameter.value.requiresGrad()) continue;
                index = collect(gradient(parameter), bucket, index, totalWeight);
            }
            if (!bucket.empty()) bucket.reduce(group, "gradient." + index, 1.0f / totalWeight);
        }
    }

    private int collect(Tensor gradient, GradientBucket bucket, int index, long totalWeight) {
        float[] source = gradient.toFloatArray();
        Tensor flat = gradient.reshape(gradient.numel());
        for (int offset = 0; offset < gradient.numel();) {
            int end = Math.min(gradient.numel(), offset + bucket.remaining());
            bucket.add(flat.slice(0, offset, end), source, offset);
            offset = end;
            if (bucket.remaining() == 0) bucket.reduce(group, "gradient." + index++, 1.0f / totalWeight);
        }
        return index;
    }

    private static Tensor gradient(Parameter parameter) {
        Tensor value = parameter.grad();
        if (value == null) value = Tensor.backend().zeros(io.github.kirstenali.deepj.tensor.DType.FLOAT32,
                parameter.value.shape());
        if (!value.isContiguous()) value = Tensor.backend().copy(value);
        parameter.setGradient(value.retainDeviceBuffer());
        return value;
    }

    void clip(List<Parameter> parameters, float maxNorm) {
        if (maxNorm == Float.POSITIVE_INFINITY) return;
        List<Tensor> gradients = parameters.stream().map(Parameter::grad).filter(java.util.Objects::nonNull).toList();
        float[] norm = {0};
        Exception failure = null;
        try {
            if (group.rank() == 0 && !gradients.isEmpty()) norm[0] = Tensor.backend().l2Norm(gradients);
            if (!Float.isFinite(norm[0])) throw new IllegalStateException("Non-finite gradient norm");
        } catch (RuntimeException error) { failure = error; }
        RankAgreement.success(group, "gradient.norm.status", failure);
        group.broadcast("gradient.norm", norm);
        if (norm[0] > maxNorm) scale(gradients, maxNorm / (norm[0] + 1e-12f));
    }

    private static void scale(List<Tensor> gradients, float scale) {
        try (var ignored = GradMode.noGrad()) {
            for (Tensor gradient : gradients) { gradient.multiplyScalarInPlace(scale); }
        }
    }
}
