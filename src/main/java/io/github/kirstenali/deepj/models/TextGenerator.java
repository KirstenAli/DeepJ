package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.models.llama.LlamaModel;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

/**
 * Autoregressive text generation for any decoder-only transformer model.
 * Supports temperature sampling and optional top-k filtering.
 */
public final class TextGenerator {

    private TextGenerator() {}

    // ── Model-specific convenience overloads ───────────────────────

    public static String generate(
            GPTModel model, Tokenizer tok, GPTConfig cfg,
            String prompt, int maxNewTokens, double temperature, int topK, long seed
    ) {
        return generate(model::forward, cfg, tok, prompt, maxNewTokens, temperature, topK, seed);
    }

    public static String generate(
            LlamaModel model, Tokenizer tok, TransformerConfig cfg,
            String prompt, int maxNewTokens, double temperature, int topK, long seed
    ) {
        return generate(model::forward, cfg, tok, prompt, maxNewTokens, temperature, topK, seed);
    }

    public static String generate(
            DeepSeekModel model, Tokenizer tok, TransformerConfig cfg,
            String prompt, int maxNewTokens, double temperature, int topK, long seed
    ) {
        return generate(model::forward, cfg, tok, prompt, maxNewTokens, temperature, topK, seed);
    }

    // ── Generic core ───────────────────────────────────────────────

    /**
     * Generate text using any model that maps {@code int[] ids → [seqLen × vocabSize]} logits.
     */
    public static String generate(
            Function<int[], Tensor> forwarder,
            TransformerConfig cfg,
            Tokenizer tok,
            String prompt,
            int maxNewTokens,
            double temperature,
            int topK,
            long seed
    ) {
        return generate(forwarder, cfg.maxSeqLen(), tok, prompt, maxNewTokens, temperature, topK, seed);
    }

    /**
     * Generate text with an explicit {@code maxSeqLen} — useful when no config is available.
     */
    public static String generate(
            Function<int[], Tensor> forwarder,
            int maxSeqLen,
            Tokenizer tok,
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
            int next = nextToken(forwarder, maxSeqLen, ids, temperature, topK, rnd);
            ids = append(ids, next);
        }

        return tok.decode(ids);
    }

    // ── Autoregressive step ────────────────────────────────────────

    private static int nextToken(Function<int[], Tensor> forwarder, int maxSeqLen, int[] ids,
                                 double temperature, int topK, Random rnd) {
        int[] context = last(ids, maxSeqLen);
        Tensor logits = forwarder.apply(context);
        double[] lastLogits = Tensor.flattenTensor(logits.getRow(logits.rows - 1));
        return sampleFromLogits(lastLogits, temperature, topK, rnd);
    }

    // ── Sampling ───────────────────────────────────────────────────

    private static int sampleFromLogits(double[] logits, double temperature, int topK, Random rnd) {
        int[] topIndices = topKIndices(logits, topK);
        double[] probs = stableSoftmax(logits, topIndices, temperature);
        return categoricalSample(topIndices, probs, rnd);
    }

    private static int[] topKIndices(double[] logits, int topK) {
        int[] order = argsortDescending(logits);
        int k = (topK == 0) ? logits.length : Math.min(topK, logits.length);
        return Arrays.copyOf(order, k);
    }

    private static double[] stableSoftmax(double[] logits, int[] indices, double temperature) {
        double max = findMaxScaledLogit(logits, indices, temperature);
        return computeProbs(logits, indices, temperature, max);
    }

    private static double findMaxScaledLogit(double[] logits, int[] indices, double temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (int idx : indices) {
            double v = logits[idx] / temperature;
            if (v > max) max = v;
        }
        return max;
    }

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
        if (topK < 0)         throw new IllegalArgumentException("topK must be >= 0");
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

