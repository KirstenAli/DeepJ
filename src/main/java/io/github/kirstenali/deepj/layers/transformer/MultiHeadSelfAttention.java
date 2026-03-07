package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.activations.Softmax;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Multi-head causal self-attention for a single sequence (no batch dimension).
 * Input/Output shape: [seqLen x dModel]

 * Notes:
 *  - This implementation is intentionally explicit (loops) to keep the code dependency-free.
 *  - For real throughput, replace inner loops with vectorized / native kernels.
 */
public final class MultiHeadSelfAttention implements Layer {

    private final int dModel;
    private final int nHeads;
    private final int headDim;
    private final boolean causalMask;

    private final Parameter Wq;
    private final Parameter Wk;
    private final Parameter Wv;
    private final Parameter Wo;

    private Tensor x;
    private Tensor Q, K, V;
    private Tensor attnProb;       // [heads*seqLen x seqLen]
    private Tensor outH;           // [heads*seqLen x headDim]
    private Tensor mergedBeforeWo; // [seqLen x dModel]

    private final ActivationFunction softmax;

    public MultiHeadSelfAttention(int dModel, int nHeads, boolean causalMask, Random rnd) {
        if (dModel % nHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by nHeads");
        }

        this.dModel = dModel;
        this.nHeads = nHeads;
        this.headDim = dModel / nHeads;
        this.causalMask = causalMask;

        this.Wq = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wk = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wv = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wo = new Parameter(Tensor.random(dModel, dModel, rnd));

        this.softmax = new Softmax();
    }

    @Override
    public Tensor forward(Tensor x) {
        validateInput(x);
        this.x = x;

        projectInputs(x);

        Tensor qh = splitHeads(Q);
        Tensor kh = splitHeads(K);
        Tensor vh = splitHeads(V);

        Tensor scores = computeScaledDotProductScores(qh, kh, x.rows);

        if (causalMask) {
            applyCausalMaskInPlace(scores, x.rows);
        }

        attnProb = softmax.forward(scores);
        outH = applyAttentionToValues(attnProb, vh, x.rows);

        mergedBeforeWo = mergeHeads(outH, x.rows);
        return mergedBeforeWo.matmul(Wo.value);
    }

    @Override
    public Tensor backward(Tensor dOut) {
        int seqLen = x.rows;

        Tensor dMerged = backwardOutputProjection(dOut);
        Tensor dOutH = splitHeads(dMerged);
        Tensor vh = splitHeads(V);

        AttentionBackwardResult attnBackward = backwardAttentionAndValues(dOutH, vh, seqLen);

        Tensor qh = splitHeads(Q);
        Tensor kh = splitHeads(K);

        QueryKeyBackwardResult qkBackward = backwardQueriesAndKeys(
                attnBackward.dScores, qh, kh, seqLen
        );

        Tensor dQ = mergeHeads(qkBackward.dQh, seqLen);
        Tensor dK = mergeHeads(qkBackward.dKh, seqLen);
        Tensor dV = mergeHeads(attnBackward.dVh, seqLen);

        accumulateProjectionGrads(dQ, dK, dV);

        return dQ.matmul(Wq.value.transpose())
                .add(dK.matmul(Wk.value.transpose()))
                .add(dV.matmul(Wv.value.transpose()));
    }

    private void validateInput(Tensor x) {
        if (x.cols != dModel) {
            throw new IllegalArgumentException("Expected cols=" + dModel + " got " + x.cols);
        }
    }

    private void projectInputs(Tensor x) {
        Q = x.matmul(Wq.value);
        K = x.matmul(Wk.value);
        V = x.matmul(Wv.value);
    }

    private Tensor computeScaledDotProductScores(Tensor qh, Tensor kh, int seqLen) {
        Tensor scores = new Tensor(nHeads * seqLen, seqLen);
        double scale = 1.0 / Math.sqrt(headDim);

        for (int h = 0; h < nHeads; h++) {
            int rowBase = h * seqLen;
            int kBase = h * seqLen;

            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    double dot = 0.0;
                    for (int d = 0; d < headDim; d++) {
                        dot += qh.data[rowBase + i][d] * kh.data[kBase + j][d];
                    }
                    scores.data[rowBase + i][j] = dot * scale;
                }
            }
        }

        return scores;
    }

    private void applyCausalMaskInPlace(Tensor scores, int seqLen) {
        Tensor mask = Tensor.causalMask(seqLen);

        for (int h = 0; h < nHeads; h++) {
            int rowBase = h * seqLen;
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    scores.data[rowBase + i][j] += mask.data[i][j];
                }
            }
        }
    }

    private Tensor applyAttentionToValues(Tensor attnProb, Tensor vh, int seqLen) {
        Tensor outH = new Tensor(nHeads * seqLen, headDim);

        for (int h = 0; h < nHeads; h++) {
            int rowBase = h * seqLen;
            int vBase = h * seqLen;

            for (int i = 0; i < seqLen; i++) {
                for (int d = 0; d < headDim; d++) {
                    double sum = 0.0;
                    for (int j = 0; j < seqLen; j++) {
                        sum += attnProb.data[rowBase + i][j] * vh.data[vBase + j][d];
                    }
                    outH.data[rowBase + i][d] = sum;
                }
            }
        }

        return outH;
    }

    private Tensor backwardOutputProjection(Tensor dOut) {
        Wo.grad = Wo.grad.add(mergedBeforeWo.transpose().matmul(dOut));
        return dOut.matmul(Wo.value.transpose());
    }

    private AttentionBackwardResult backwardAttentionAndValues(Tensor dOutH, Tensor vh, int seqLen) {
        Tensor dAttn = new Tensor(nHeads * seqLen, seqLen);
        Tensor dVh = new Tensor(nHeads * seqLen, headDim);

        for (int h = 0; h < nHeads; h++) {
            int rowBase = h * seqLen;
            int vBase = h * seqLen;

            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    double sum = 0.0;
                    for (int d = 0; d < headDim; d++) {
                        sum += dOutH.data[rowBase + i][d] * vh.data[vBase + j][d];
                    }
                    dAttn.data[rowBase + i][j] = sum;
                }

                for (int j = 0; j < seqLen; j++) {
                    for (int d = 0; d < headDim; d++) {
                        dVh.data[vBase + j][d] += attnProb.data[rowBase + i][j] * dOutH.data[rowBase + i][d];
                    }
                }
            }
        }

        Tensor dScores = softmax.backward(dAttn)
                .multiplyScalar(1.0 / Math.sqrt(headDim));

        return new AttentionBackwardResult(dScores, dVh);
    }

    private QueryKeyBackwardResult backwardQueriesAndKeys(
            Tensor dScores, Tensor qh, Tensor kh, int seqLen
    ) {
        Tensor dQh = new Tensor(nHeads * seqLen, headDim);
        Tensor dKh = new Tensor(nHeads * seqLen, headDim);

        for (int h = 0; h < nHeads; h++) {
            int rowBase = h * seqLen;
            int kBase = h * seqLen;

            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    double g = dScores.data[rowBase + i][j];
                    for (int d = 0; d < headDim; d++) {
                        dQh.data[rowBase + i][d] += g * kh.data[kBase + j][d];
                        dKh.data[kBase + j][d] += g * qh.data[rowBase + i][d];
                    }
                }
            }
        }

        return new QueryKeyBackwardResult(dQh, dKh);
    }

    private void accumulateProjectionGrads(Tensor dQ, Tensor dK, Tensor dV) {
        Wq.grad = Wq.grad.add(x.transpose().matmul(dQ));
        Wk.grad = Wk.grad.add(x.transpose().matmul(dK));
        Wv.grad = Wv.grad.add(x.transpose().matmul(dV));
    }

    private Tensor splitHeads(Tensor t) {
        int seqLen = t.rows;
        Tensor out = new Tensor(nHeads * seqLen, headDim);

        for (int i = 0; i < seqLen; i++) {
            for (int h = 0; h < nHeads; h++) {
                int row = h * seqLen + i;
                int colBase = h * headDim;
                System.arraycopy(t.data[i], colBase, out.data[row], 0, headDim);
            }
        }
        return out;
    }

    private Tensor mergeHeads(Tensor t, int seqLen) {
        Tensor out = new Tensor(seqLen, dModel);

        for (int i = 0; i < seqLen; i++) {
            for (int h = 0; h < nHeads; h++) {
                int row = h * seqLen + i;
                int colBase = h * headDim;
                System.arraycopy(t.data[row], 0, out.data[i], colBase, headDim);
            }
        }

        return out;
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        ps.add(Wq);
        ps.add(Wk);
        ps.add(Wv);
        ps.add(Wo);
        return ps;
    }

    private record AttentionBackwardResult(Tensor dScores, Tensor dVh) {
    }

    private record QueryKeyBackwardResult(Tensor dQh, Tensor dKh) {
    }
}