package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.inference.InferenceContext;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.GenerationSession;
import io.github.kirstenali.deepj.tensor.BackwardBuffers;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Consumer;

final class Qwen3Session implements GenerationSession {
    private static final int INITIAL_CAPACITY = 128;

    private final Qwen3Config config;
    private final BiFunction<int[], InferenceContext, Tensor> forward;
    private final Consumer<int[]> validateTokens;
    private final TensorBackend backend = Tensor.backend();
    private InferenceContext cache;
    private final Map<Tensor, Long> weights = new IdentityHashMap<>();
    private final boolean rolling;
    private int[] history = new int[0];
    private int position, capacity;
    private boolean closed;

    Qwen3Session(Qwen3 model, ContextWindowMode mode) {
        this(model.config(), model.parameters(), model::cached, model::requireLength, mode);
    }

    Qwen3Session(Qwen3Config config, List<Parameter> parameters, BiFunction<int[], InferenceContext, Tensor> forward,
                 Consumer<int[]> validateTokens, ContextWindowMode mode) {
        this.config = config;
        this.forward = forward;
        this.validateTokens = validateTokens;
        rolling = mode == ContextWindowMode.ROLLING;
        capacity = Math.min(INITIAL_CAPACITY, config.maxSeqLen());
        cache = new InferenceContext(capacity, rolling);
        parameters.forEach(parameter ->
                weights.put(parameter.value, parameter.value.gradientState().version()));
    }

    @Override public Tensor apply(int[] tokens) {
        validate(tokens);
        boolean continuation = prepare(tokens, appended(tokens) || shifted(tokens));
        cache.begin(position);
        try (var ignored = GradMode.noGrad()) {
            int[] input = continuation ? new int[]{tokens[tokens.length - 1]} : tokens;
            Tensor result = cached(input);
            cache.finish();
            position = Math.addExact(position, input.length);
            history = tokens.clone();
            return result;
        } catch (RuntimeException | Error failure) {
            close();
            throw failure;
        }
    }

    private Tensor cached(int[] input) {
        if (!backend.usesDeviceStorage()) return forward.apply(input, cache);
        try (var scratch = BackwardBuffers.begin()) {
            Tensor logits = scratch.capture(() -> forward.apply(input, cache));
            scratch.expose(logits);
            return logits;
        }
    }

    private boolean prepare(int[] tokens, boolean continuation) {
        if (continuation && tokens.length <= capacity) return true;
        if (continuation) {
            int target = (int) Math.min(config.maxSeqLen(), Math.max(tokens.length, 2L * capacity));
            InferenceContext previous = cache;
            cache = cache.grown(target);
            previous.close();
            capacity = target;
            return true;
        }
        reset();
        long requested = promptCapacity(tokens.length);
        capacity = (int) Math.min(config.maxSeqLen(), requested);
        cache = new InferenceContext(capacity, rolling);
        return false;
    }

    private static long promptCapacity(int length) {
        if (length < INITIAL_CAPACITY) return INITIAL_CAPACITY;
        return (long) length + INITIAL_CAPACITY;
    }

    private void validate(int[] tokens) {
        if (closed || backend != Tensor.backend()) throw new IllegalStateException("Generation session is unavailable");
        validateTokens.accept(tokens);
        weights.forEach((tensor, version) -> {
            if (tensor.gradientState().version() != version) {
                throw new IllegalStateException("Weights changed; open a new generation session");
            }
        });
    }

    private boolean appended(int[] tokens) {
        return history.length > 0 && tokens.length - history.length == 1
                && Arrays.equals(history, 0, history.length, tokens, 0, history.length);
    }

    private boolean shifted(int[] tokens) {
        return rolling && history.length == config.maxSeqLen() && tokens.length == history.length
                && Arrays.equals(history, 1, history.length, tokens, 0, tokens.length - 1);
    }

    private void reset() {
        cache.close();
        history = new int[0];
        position = 0;
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        reset();
    }
}
