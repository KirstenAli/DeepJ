package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.inference.InferenceContext;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;

import java.util.Arrays;
import java.util.List;

final class CachedGenerationSession implements GenerationSession {
    private final DecoderOnlyModel model;
    private final TensorBackend backend = Tensor.backend();
    private final InferenceContext context;
    private final List<Parameter> parameters;
    private final long[] versions;
    private int[] previous = new int[0];
    private boolean closed;

    CachedGenerationSession(DecoderOnlyModel model) {
        this.model = model;
        context = new InferenceContext(model.config().maxSeqLen());
        parameters = model.parameters();
        versions = parameters.stream().mapToLong(p -> p.value.gradientState().version()).toArray();
    }

    @Override
    public Tensor apply(int[] tokens) {
        requireUsable(tokens);
        int offset = extendsPrevious(tokens) ? previous.length : 0;
        if (offset == 0) reset();
        context.begin(offset);
        try (var ignored = GradMode.noGrad()) {
            Tensor logits = model.forwardCached(offset == 0 ? tokens : new int[]{tokens[offset]}, context);
            context.finish();
            previous = tokens.clone();
            return logits;
        } catch (RuntimeException | Error failure) {
            close();
            throw failure;
        }
    }

    private boolean extendsPrevious(int[] tokens) {
        return previous.length > 0 && tokens.length == previous.length + 1
                && Arrays.equals(previous, 0, previous.length, tokens, 0, previous.length);
    }

    private void requireUsable(int[] tokens) {
        if (closed) throw new IllegalStateException("Generation session is closed");
        if (backend != Tensor.backend()) throw new IllegalStateException("Generation backend changed");
        if (tokens == null || tokens.length < 1 || tokens.length > model.config().maxSeqLen()) {
            throw new IllegalArgumentException("Generation context is empty or exceeds the model limit");
        }
        requireUnchangedWeights();
    }

    private void requireUnchangedWeights() {
        for (int index = 0; index < versions.length; index++) {
            if (versions[index] != parameters.get(index).value.gradientState().version()) {
                throw new IllegalStateException("Model weights changed during generation; open a new session");
            }
        }
    }

    private void reset() {
        context.close();
        previous = new int[0];
        backend.releaseTemporaryResources();
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        reset();
    }
}
