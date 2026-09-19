package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class AdamW implements ParameterOptimizer {

    private float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;

    private long step = 0;

    private final Map<Parameter, Tensor> m = new IdentityHashMap<>();
    private final Map<Parameter, Tensor> v = new IdentityHashMap<>();

    public AdamW(float lr, float beta1, float beta2, float eps, float weightDecay) {
        validateHyperparameters(lr, beta1, beta2, eps);

        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
    }

    public static AdamW defaultAdamW(float lr) {
        return new AdamW(lr, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    public float lr() {
        return lr;
    }

    public void setLr(float lr) {
        if (lr <= 0) {
            throw new IllegalArgumentException("lr must be > 0");
        }
        this.lr = lr;
    }

    public long stepCount() {
        return step;
    }

    public State state(List<Parameter> params) {
        Objects.requireNonNull(params, "params");
        return new State(step, lr, beta1, beta2, eps, weightDecay,
                moments(params, m), moments(params, v));
    }

    public void restoreState(List<Parameter> params, State state) {
        Objects.requireNonNull(params, "params");
        Objects.requireNonNull(state, "state");
        validateState(params, state);
        m.clear();
        v.clear();
        for (int index = 0; index < params.size(); index++) {
            m.put(params.get(index), retained(state.firstMoments().get(index)));
            v.put(params.get(index), retained(state.secondMoments().get(index)));
        }
        step = state.step();
        lr = state.learningRate();
    }

    private static List<Tensor> moments(List<Parameter> params, Map<Parameter, Tensor> source) {
        return params.stream().map(parameter -> moment(parameter, source)).toList();
    }

    private static Tensor moment(Parameter parameter, Map<Parameter, Tensor> source) {
        Tensor saved = source.get(parameter);
        return saved == null ? newMoment(parameter.value) : saved;
    }

    private static Tensor newMoment(Tensor value) {
        return Tensor.zeros(value.rows, value.cols).retainDeviceBuffer();
    }

    private static Tensor retained(Tensor tensor) {
        return tensor.retainDeviceBuffer();
    }

    private void validateState(List<Parameter> params, State state) {
        if (state.step() < 0 || state.learningRate() <= 0) {
            throw new IllegalArgumentException("invalid AdamW state counters");
        }
        validateStateHyperparameters(state);
        validateMomentList(params, state.firstMoments());
        validateMomentList(params, state.secondMoments());
    }

    private void validateStateHyperparameters(State state) {
        if (Float.compare(beta1, state.beta1()) != 0
                || Float.compare(beta2, state.beta2()) != 0
                || Float.compare(eps, state.epsilon()) != 0
                || Float.compare(weightDecay, state.weightDecay()) != 0) {
            throw new IllegalArgumentException("AdamW hyperparameters do not match checkpoint");
        }
    }

    private static void validateMomentList(List<Parameter> params, List<Tensor> moments) {
        if (moments == null || moments.size() != params.size()) {
            throw new IllegalArgumentException("AdamW moment count does not match parameters");
        }
        for (int index = 0; index < params.size(); index++) {
            Tensor value = params.get(index).value;
            Tensor moment = Objects.requireNonNull(moments.get(index), "moment");
            if (value.rows != moment.rows || value.cols != moment.cols) {
                throw new IllegalArgumentException("AdamW moment shape does not match parameter");
            }
        }
    }

    @Override
    public void step(List<Parameter> params) {
        if (params == null) {
            throw new IllegalArgumentException("params must not be null");
        }

        step++;

        float bc1 = 1.0f - (float) Math.pow(beta1, step);
        float bc2 = 1.0f - (float) Math.pow(beta2, step);

        for (Parameter p : params) {
            if (p != null) {
                stepParam(p, bc1, bc2);
            }
        }
    }

    private void validateHyperparameters(float lr, float beta1, float beta2, float eps) {
        if (lr <= 0) {
            throw new IllegalArgumentException("lr must be > 0");
        }
        if (beta1 <= 0 || beta1 >= 1) {
            throw new IllegalArgumentException("beta1 must be in (0,1)");
        }
        if (beta2 <= 0 || beta2 >= 1) {
            throw new IllegalArgumentException("beta2 must be in (0,1)");
        }
        if (eps <= 0) {
            throw new IllegalArgumentException("eps must be > 0");
        }
    }

    private void stepParam(Parameter p, float bc1, float bc2) {
        Tensor w = p.value;
        Tensor g = p.grad;

        if (w == null || g == null) {
            return;
        }

        validateParamShapes(w, g);

        Tensor mt = m.computeIfAbsent(p, __ -> newMoment(w));
        Tensor vt = v.computeIfAbsent(p, __ -> newMoment(w));

        Tensor.adamWUpdate(w, g, mt, vt, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
    }

    private void validateParamShapes(Tensor w, Tensor g) {
        if (w.rows != g.rows || w.cols != g.cols) {
            throw new IllegalArgumentException("grad shape must match param shape");
        }
    }

    public record State(long step, float learningRate, float beta1, float beta2,
                        float epsilon, float weightDecay, List<Tensor> firstMoments,
                        List<Tensor> secondMoments) {

        public State {
            firstMoments = List.copyOf(firstMoments);
            secondMoments = List.copyOf(secondMoments);
        }
    }
}
