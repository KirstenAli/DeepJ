package io.github.kirstenali.deepj.tensor;

/** Final-axis normalization accepts vector gains and the historical 1-by-width parameter layout. */
public final class NormalizationChecks {
    private NormalizationChecks() {}

    public static int width(Tensor input) {
        if (input.rank() == 0 || input.size(-1) == 0) {
            throw new IllegalArgumentException("Normalization requires a nonempty final axis");
        }
        return input.size(-1);
    }

    public static void parameter(Tensor parameter, int width) {
        if (parameter.rank() != 1 && !(parameter.rank() == 2 && parameter.size(0) == 1)) {
            throw new IllegalArgumentException("Normalization parameters must be vectors or 1-by-width matrices");
        }
        if (parameter.size(-1) != width) throw new IllegalArgumentException("Normalization parameter width mismatch");
    }

    public static int[] statisticsShape(Tensor input) {
        int[] shape = input.shape();
        shape[shape.length - 1] = 1;
        return shape;
    }

    public static DType outputType(Tensor input, Tensor gamma, Tensor beta, boolean centered) {
        DType type = DType.promote(input.dtype(), gamma.dtype());
        return centered ? DType.promote(type, beta.dtype()) : type;
    }

    public static void forward(Tensor input, Tensor gamma, Tensor beta, float epsilon, boolean centered) {
        int width = width(input);
        parameter(gamma, width);
        if (centered) parameter(java.util.Objects.requireNonNull(beta, "beta"), width);
        else if (beta != null) throw new IllegalArgumentException("RMS normalization has no bias");
        if (!Float.isFinite(epsilon) || epsilon <= 0) {
            throw new IllegalArgumentException("Normalization epsilon must be finite and positive");
        }
    }

    public static void backward(Tensor gradient, Tensor normalized, Tensor denominator, Tensor gamma) {
        parameter(gamma, width(normalized));
        Tensor.requireSameShape(gradient, normalized, "Normalization gradient");
        statistics(normalized, denominator);
    }

    public static void statistics(Tensor input, Tensor denominator) {
        width(input);
        if (!denominator.layout().shape().equals(new TensorShape(statisticsShape(input)))) {
            throw new IllegalArgumentException("Normalization denominator shape mismatch");
        }
    }
}
