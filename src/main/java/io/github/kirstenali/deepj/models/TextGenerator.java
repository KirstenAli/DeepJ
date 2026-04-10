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
            String prompt, int maxNewTokens, float temperature, int topK, long seed
    ) {
        return generate(model::forward, cfg, tok, prompt, maxNewTokens, temperature, topK, seed);
    }

    public static String generate(
            LlamaModel model, Tokenizer tok, TransformerConfig cfg,
            String prompt, int maxNewTokens, float temperature, int topK, long seed
    ) {
        return generate(model::forward, cfg, tok, prompt, maxNewTokens, temperature, topK, seed);
    }

    public static String generate(
            DeepSeekModel model, Tokenizer tok, TransformerConfig cfg,
            String prompt, int maxNewTokens, float temperature, int topK, long seed
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
            float temperature,
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
            float temperature,
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
                                 float temperature, int topK, Random rnd) {
        int[] context = last(ids, maxSeqLen);
        Tensor logits = forwarder.apply(context);
        logits.materialize();
        // Extract last row directly from flat storage — no backend round-trip.
        float[] lastLogits = Arrays.copyOfRange(logits.data,
                (logits.rows - 1) * logits.cols, logits.rows * logits.cols);
        return sampleFromLogits(lastLogits, temperature, topK, rnd);
    }

    // ── Sampling ───────────────────────────────────────────────────

    private static int sampleFromLogits(float[] logits, float temperature, int topK, Random rnd) {
        int[] topIndices = topKIndices(logits, topK);
        float[] probs = stableSoftmax(logits, topIndices, temperature);
        return categoricalSample(topIndices, probs, rnd);
    }

    private static int[] topKIndices(float[] logits, int topK) {
        int[] order = argsortDescending(logits);
        int k = (topK == 0) ? logits.length : Math.min(topK, logits.length);
        return Arrays.copyOf(order, k);
    }

    private static float[] stableSoftmax(float[] logits, int[] indices, float temperature) {
        float max = findMaxScaledLogit(logits, indices, temperature);
        return computeProbs(logits, indices, temperature, max);
    }

    private static float findMaxScaledLogit(float[] logits, int[] indices, float temperature) {
        float max = Float.NEGATIVE_INFINITY;
        for (int idx : indices) {
            float v = logits[idx] / temperature;
            if (v > max) max = v;
        }
        return max;
    }

    private static float[] computeProbs(float[] logits, int[] indices,
                                         float temperature, float max) {
        float[] probs = new float[indices.length];
        float sum = 0.0f;
        for (int i = 0; i < indices.length; i++) {
            float p = (float) Math.exp(logits[indices[i]] / temperature - max);
            probs[i] = p;
            sum += p;
        }
        for (int i = 0; i < probs.length; i++) probs[i] /= sum;
        return probs;
    }

    private static int categoricalSample(int[] indices, float[] probs, Random rnd) {
        float r = rnd.nextFloat();
        float cum = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r <= cum) return indices[i];
        }
        return indices[indices.length - 1];
    }

    // ── Validation ─────────────────────────────────────────────────

    private static void validateArgs(int maxNewTokens, float temperature, int topK) {
        if (maxNewTokens < 0) throw new IllegalArgumentException("maxNewTokens must be >= 0");
        if (temperature <= 0.0f) throw new IllegalArgumentException("temperature must be > 0");
        if (topK < 0)         throw new IllegalArgumentException("topK must be >= 0");
    }

    // ── Array utilities ────────────────────────────────────────────

    private static int[] argsortDescending(float[] a) {
        Integer[] idx = new Integer[a.length];
        for (int i = 0; i < a.length; i++) idx[i] = i;
        Arrays.sort(idx, (i, j) -> Float.compare(a[j], a[i]));
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

