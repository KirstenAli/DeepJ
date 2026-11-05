package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.transformer.GenerationSession;
import io.github.kirstenali.deepj.tensor.BackwardBuffers;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.Map;

/** A request owns its KV cache; weight mutation, changed prompts and context rollover invalidate it. */
final class Gemma2Session implements GenerationSession {
    private final Gemma2 model;
    private final TensorBackend backend = Tensor.backend();
    private final Map<Tensor, Long> versions = new IdentityHashMap<>();
    private Gemma2Cache cache;
    private int[] history = new int[0];
    private int capacity;
    private boolean closed;

    Gemma2Session(Gemma2 model) {
        this.model = model;
        model.parameters().forEach(parameter ->
                versions.put(parameter.value, parameter.value.gradientState().version()));
    }

    @Override public Tensor apply(int[] tokens) {
        validate(tokens);
        boolean appended = history.length > 0 && tokens.length == history.length + 1
                && Arrays.equals(history, 0, history.length, tokens, 0, history.length);
        if (!appended || tokens.length > capacity) reset(tokens.length);
        int[] input = appended && history.length > 0 ? new int[]{tokens[tokens.length - 1]} : tokens;
        try (var ignored = GradMode.noGrad()) {
            Tensor result = evaluate(input);
            cache.finish(input.length);
            history = tokens.clone();
            return result;
        } catch (RuntimeException | Error failure) {
            close();
            throw failure;
        }
    }

    private Tensor evaluate(int[] input) {
        if (input.length <= 1 || !backend.usesDeviceStorage()) return model.logits(input, cache);
        try (var scratch = BackwardBuffers.begin()) {
            Tensor result = scratch.capture(() -> model.logits(input, cache));
            scratch.expose(result);
            return result;
        }
    }

    private void validate(int[] tokens) {
        if (closed || backend != Tensor.backend()) throw new IllegalStateException("Gemma session is unavailable");
        if (tokens.length < 1 || tokens.length > model.config().maxSeqLen()) {
            throw new IllegalArgumentException("Gemma context length is outside the model window");
        }
        versions.forEach((tensor, version) -> {
            if (tensor.gradientState().version() != version) {
                throw new IllegalStateException("Gemma weights changed; open a new session");
            }
        });
    }

    private void reset(int length) {
        if (cache != null) cache.close();
        capacity = (int) Math.min(model.config().maxSeqLen(), Math.max(length, Math.max(128L, capacity * 2L)));
        cache = new Gemma2Cache(model.config(), capacity);
        history = new int[0];
    }

    @Override public void close() {
        if (cache != null) cache.close();
        cache = null;
        closed = true;
    }
}
