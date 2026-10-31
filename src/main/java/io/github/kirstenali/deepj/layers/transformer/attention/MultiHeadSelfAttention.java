package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.activations.Softmax;
import io.github.kirstenali.deepj.inference.InferenceContext;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.ProjectionOps;
import io.github.kirstenali.deepj.tensor.SequenceLayout;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;

import java.util.List;
import java.util.Random;

public class MultiHeadSelfAttention implements Layer {

    private final int dModel;

    protected final int nHeads;

    protected final int headDim;
    private final boolean causalMask;
    private final float scale;

    private final Parameter Wq;
    private final Parameter Wk;
    private final Parameter Wv;
    private final Parameter Wo;

    private final Softmax softmax;

    public MultiHeadSelfAttention(int dModel, int nHeads, boolean causalMask, Random rnd) {
        this(dModel, nHeads, causalMask, rnd, ParameterFactory.float32());
    }

    public MultiHeadSelfAttention(int dModel, int nHeads, boolean causalMask, Random rnd,
                                   ParameterFactory parameters) {
        validateDimensions(dModel, nHeads);

        this.dModel = dModel;
        this.nHeads = nHeads;
        this.headDim = dModel / nHeads;
        this.causalMask = causalMask;
        this.scale = (float) (1.0 / Math.sqrt(headDim));

        this.Wq = parameters.random("query.weight", dModel, dModel, rnd);
        this.Wk = parameters.random("key.weight", dModel, dModel, rnd);
        this.Wv = parameters.random("value.weight", dModel, dModel, rnd);
        this.Wo = parameters.random("output.weight", dModel, dModel, rnd);

        this.softmax = new Softmax();
    }

    private static void validateDimensions(int dModel, int nHeads) {
        if (dModel <= 0) throw new IllegalArgumentException("dModel must be > 0");
        if (nHeads <= 0) throw new IllegalArgumentException("nHeads must be > 0");
        if (dModel % nHeads != 0) throw new IllegalArgumentException("dModel must be divisible by nHeads");
    }

    @Override
    public Tensor forward(Tensor x) {
        validateInput(x);
        Projections proj = projectInputs(x);
        Tensor queries = transformQueryKey(HeadOps.splitHeads(proj.Q, nHeads), SequenceLayout.length(x));
        Tensor keys = transformQueryKey(HeadOps.splitHeads(proj.K, nHeads), SequenceLayout.length(x));
        Tensor values = HeadOps.splitHeads(proj.V, nHeads);
        Tensor context = attend(queries, keys, values);
        return ProjectionOps.linear(HeadOps.mergeHeads(context, nHeads), Wo.value);
    }

    protected Tensor transformQueryKey(Tensor heads, int seqLen) {
        return heads;
    }

    protected Tensor transformQueryKey(Tensor heads, int length, int offset) {
        return transformQueryKey(heads, length);
    }

    protected Tensor transformQueryKey(Tensor heads, int length, InferenceContext context) {
        return transformQueryKey(heads, length, context.position());
    }

    @Override
    public boolean supportsIncrementalInference() { return causalMask; }

    @Override
    public Tensor forward(Tensor input, InferenceContext context) {
        if (!causalMask) throw new IllegalStateException("Noncausal attention cannot use a KV cache");
        validateInput(input);
        Projections projections = projectInputs(input);
        Tensor queries = transformQueryKey(HeadOps.splitHeads(projections.Q, nHeads),
                SequenceLayout.length(input), context);
        Tensor keys = transformQueryKey(HeadOps.splitHeads(projections.K, nHeads),
                SequenceLayout.length(input), context);
        Tensor values = HeadOps.splitHeads(projections.V, nHeads);
        Tensor attended = context.attend(this, queries, keys, values, nHeads, nHeads, scale);
        return ProjectionOps.linear(HeadOps.mergeHeads(attended, nHeads), Wo.value);
    }

    private void validateInput(Tensor x) {
        SequenceLayout.requireWidth(x, dModel);
    }

    private Projections projectInputs(Tensor x) {
        return new Projections(
                ProjectionOps.linear(x, Wq.value),
                ProjectionOps.linear(x, Wk.value),
                ProjectionOps.linear(x, Wv.value));
    }

    private Tensor attend(Tensor queries, Tensor keys, Tensor values) {
        if (causalMask) return HeadOps.causal(queries, keys, values, nHeads, scale);
        Tensor scores = HeadOps.dotProductScores(queries, keys, nHeads);
        return HeadOps.applyAttentionToValues(softmax.forward(scores.multiplyScalar(scale)), values, nHeads);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(Wq, Wk, Wv, Wo);
    }

    private record Projections(Tensor Q, Tensor K, Tensor V) {}
}
