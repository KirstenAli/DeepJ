package io.github.kirstenali.deepj.models.gpt;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.util.Arrays;
import java.util.Random;

/**
 * Simple autoregressive text generation for {@link GPTModel}.
 * Byte-level tokens, temperature sampling, and optional top-k sampling.
 */
public final class TextGenerator {

    private TextGenerator() {}

    // ── Public API ─────────────────────────────────────────────────

    public static String generate(
            GPTModel model,
            Tokenizer tok,
            GPTConfig cfg,
            String prompt,
            int maxNewTokens,
            double temperature,
            int topK,
            long seed
    ) {
        validateArgs(maxNewTokens, temperature, topK);

        Random rnd = new Random(seed);
        int[] ids = tok.encode(prompt).clone();

        for (int i = 0; i < maxNewTokens; i++) {
            int next = nextToken(model, cfg, ids, temperature, topK, rnd);
            ids = append(ids, next);
        }

        return tok.decode(ids);
    }

    // ── Autoregressive step ────────────────────────────────────────

    /** Run one forward pass and sample the next token from the last position's logits. */
    private static int nextToken(GPTModel model, GPTConfig cfg, int[] ids,
                                 double temperature, int topK, Random rnd) {
        int[] context = last(ids, cfg.maxSeqLen());
        Tensor logits = model.forward(context);                    // [seqLen x vocab]
        double[] lastLogits = Tensor.flattenTensor(logits.getRow(logits.rows - 1));
        return sampleFromLogits(lastLogits, temperature, topK, rnd);
    }

    // ── Sampling ───────────────────────────────────────────────────

    /** Apply temperature, top-k filtering, and softmax, then draw one token. */
    private static int sampleFromLogits(double[] logits, double temperature, int topK, Random rnd) {
        int[] topIndices = topKIndices(logits, topK);
        double[] probs = stableSoftmax(logits, topIndices, temperature);
        return categoricalSample(topIndices, probs, rnd);
    }

    /** Return the indices of the top-k logits in descending order (all indices if topK == 0). */
    private static int[] topKIndices(double[] logits, int topK) {
        int[] order = argsortDescending(logits);
        int k = (topK == 0) ? logits.length : Math.min(topK, logits.length);
        return Arrays.copyOf(order, k);
    }

    /** Compute a numerically-stable softmax over the selected indices with temperature scaling. */
    private static double[] stableSoftmax(double[] logits, int[] indices, double temperature) {
        double max = findMaxScaledLogit(logits, indices, temperature);
        return computeProbs(logits, indices, temperature, max);
    }

    /** Find the maximum temperature-scaled logit among the selected indices. */
    private static double findMaxScaledLogit(double[] logits, int[] indices, double temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (int idx : indices) {
            double v = logits[idx] / temperature;
            if (v > max) max = v;
        }
        return max;
    }

    /** Exponentiate, accumulate sum, and normalize in two passes (same as original). */
    private static double[] computeProbs(double[] logits, int[] indices,
                                         double temperature, double max) {
        double[] probs = new double[indices.length];
        double sum = 0.0;
        for (int i = 0; i < indices.length; i++) {
            double p = Math.exp(logits[indices[i]] / temperature - max);
            probs[i] = p;
            sum += p;
        }
        for (int i = 0; i < probs.length; i++) probs[i] /= sum;
        return probs;
    }


    /** Draw one index from a categorical distribution defined by probs. */
    private static int categoricalSample(int[] indices, double[] probs, Random rnd) {
        double r = rnd.nextDouble();
        double cum = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r <= cum) return indices[i];
        }
        return indices[indices.length - 1];
    }

    // ── Validation ─────────────────────────────────────────────────

    private static void validateArgs(int maxNewTokens, double temperature, int topK) {
        if (maxNewTokens < 0) throw new IllegalArgumentException("maxNewTokens must be >= 0");
        if (temperature <= 0) throw new IllegalArgumentException("temperature must be > 0");
        if (topK < 0) throw new IllegalArgumentException("topK must be >= 0");
    }

    // ── Array utilities ────────────────────────────────────────────

    private static int[] argsortDescending(double[] a) {
        Integer[] idx = new Integer[a.length];
        for (int i = 0; i < a.length; i++) idx[i] = i;
        Arrays.sort(idx, (i, j) -> Double.compare(a[j], a[i]));
        int[] out = new int[a.length];
        for (int i = 0; i < a.length; i++) out[i] = idx[i];
        return out;
    }

    private static int[] last(int[] a, int n) {
        if (a.length <= n) return a;
        return Arrays.copyOfRange(a, a.length - n, a.length);
    }

    private static int[] append(int[] a, int v) {
        int[] out = Arrays.copyOf(a, a.length + 1);
        out[a.length] = v;
        return out;
    }
}
