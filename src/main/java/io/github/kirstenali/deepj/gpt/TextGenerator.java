package io.github.kirstenali.deepj.gpt;

import io.github.kirstenali.deepj.tokenizer.ByteTokenizer;

import io.github.kirstenali.deepj.Tensor;

import java.util.Arrays;
import java.util.Random;

/**
 * Simple autoregressive text generation for {@link GPTModel}.
 * Byte-level tokens, temperature sampling, and optional top-k sampling.
 */
public final class TextGenerator {

    private TextGenerator() {}

    public static String generate(
            GPTModel model,
            ByteTokenizer tok,
            GPTConfig cfg,
            String prompt,
            int maxNewTokens,
            double temperature,
            int topK,
            long seed
    ) {
        if (maxNewTokens < 0) throw new IllegalArgumentException("maxNewTokens must be >= 0");
        if (temperature <= 0) throw new IllegalArgumentException("temperature must be > 0");
        if (topK < 0) throw new IllegalArgumentException("topK must be >= 0");

        Random rnd = new Random(seed);

        int[] promptIds = tok.encode(prompt);
        int[] ids = Arrays.copyOf(promptIds, promptIds.length);

        for (int i = 0; i < maxNewTokens; i++) {
            int[] context = last(ids, cfg.maxSeqLen());
            Tensor logits = model.forward(context); // [seqLen x vocab]
            double[] lastLogits = logits.data[logits.rows - 1];
            int next = sampleFromLogits(lastLogits, temperature, topK, rnd);
            ids = append(ids, next);
        }

        return tok.decode(ids);
    }

    private static int sampleFromLogits(double[] logits, double temperature, int topK, Random rnd) {
        int[] order = argsortDescending(logits);
        int k = (topK == 0) ? logits.length : Math.min(topK, logits.length);

        // Stable softmax over the top-k subset
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < k; i++) {
            double v = logits[order[i]] / temperature;
            if (v > max) max = v;
        }

        double sum = 0.0;
        double[] probs = new double[k];
        for (int i = 0; i < k; i++) {
            double v = logits[order[i]] / temperature;
            double p = Math.exp(v - max);
            probs[i] = p;
            sum += p;
        }

        double r = rnd.nextDouble() * sum;
        double cum = 0.0;
        for (int i = 0; i < k; i++) {
            cum += probs[i];
            if (r <= cum) return order[i];
        }
        return order[k - 1];
    }

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
