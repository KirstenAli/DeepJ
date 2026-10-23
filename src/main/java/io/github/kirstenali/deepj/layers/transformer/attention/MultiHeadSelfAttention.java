package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.activations.Softmax;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;

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
        validateDimensions(dModel, nHeads);

        this.dModel = dModel;
        this.nHeads = nHeads;
        this.headDim = dModel / nHeads;
        this.causalMask = causalMask;
        this.scale = (float) (1.0 / Math.sqrt(headDim));

        this.Wq = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wk = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wv = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wo = new Parameter(Tensor.random(dModel, dModel, rnd));

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
        Tensor queries = transformQueryKey(HeadOps.splitHeads(proj.Q, nHeads), x.rows);
        Tensor keys = transformQueryKey(HeadOps.splitHeads(proj.K, nHeads), x.rows);
        Tensor values = HeadOps.splitHeads(proj.V, nHeads);
        Tensor context = attend(queries, keys, values);
        return HeadOps.mergeHeads(context, nHeads).matmul(Wo.value);
    }

    protected Tensor transformQueryKey(Tensor heads, int seqLen) {
        return heads;
    }

    private void validateInput(Tensor x) {
        if (x.cols != dModel) {
            throw new IllegalArgumentException("Expected cols=" + dModel + " got " + x.cols);
        }
    }

    private Projections projectInputs(Tensor x) {
        return new Projections(
                x.matmul(Wq.value),
                x.matmul(Wk.value),
                x.matmul(Wv.value));
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
