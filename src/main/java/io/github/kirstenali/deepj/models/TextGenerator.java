package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbit;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public final class TextGenerator {

    private TextGenerator() {}

    public static String generate(
            DeepJOrigin model, Tokenizer tok, DeepJOriginConfig cfg,
            String prompt, int maxNewTokens, float temperature, int topK, long seed
    ) {
        return generate(model::forward, cfg, tok, prompt, maxNewTokens, temperature, topK, seed);
    }

    public static String generate(
            DeepJOrbit model, Tokenizer tok, TransformerConfig cfg,
            String prompt, int maxNewTokens, float temperature, int topK, long seed
    ) {
        return generate(model::forward, cfg, tok, prompt, maxNewTokens, temperature, topK, seed);
    }

    public static String generate(
            DeepJPrism model, Tokenizer tok, TransformerConfig cfg,
            String prompt, int maxNewTokens, float temperature, int topK, long seed
    ) {
        return generate(model::forward, cfg, tok, prompt, maxNewTokens, temperature, topK, seed);
    }

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
        int[] promptIds = tok.encode(prompt);
        requirePrompt(promptIds, maxNewTokens);
        int[] ids = Arrays.copyOf(promptIds, promptIds.length + maxNewTokens);
        int length = appendGeneratedTokens(forwarder, maxSeqLen, tok, ids,
                promptIds.length, maxNewTokens, temperature, topK, rnd);
        return tok.decode(Arrays.copyOf(ids, length));
    }

    private static int appendGeneratedTokens(Function<int[], Tensor> forwarder, int maxSeqLen,
                                             Tokenizer tok, int[] ids, int length,
                                             int maxNewTokens, float temperature, int topK,
                                             Random rnd) {
        if (hasTrailingEndToken(tok, ids, length)) return length;
        for (int i = 0; i < maxNewTokens; i++) {
            int token = nextToken(forwarder, maxSeqLen, ids, length, temperature, topK, rnd);
            if (tok.isEndOfSequence(token)) break;
            ids[length++] = token;
        }
        return length;
    }

    private static boolean hasTrailingEndToken(Tokenizer tokenizer, int[] ids, int length) {
        return length > 0 && tokenizer.isEndOfSequence(ids[length - 1]);
    }

    private static int nextToken(Function<int[], Tensor> forwarder, int maxSeqLen,
                                 int[] ids, int length, float temperature, int topK, Random rnd) {
        int[] context = context(ids, length, maxSeqLen);
        Tensor logits = forwarder.apply(context);
        logits.materialize();

        float[] lastLogits = Arrays.copyOfRange(logits.data,
                (logits.rows - 1) * logits.cols, logits.rows * logits.cols);
        return sampleFromLogits(lastLogits, temperature, topK, rnd);
    }

    private static int sampleFromLogits(float[] logits, float temperature, int topK, Random rnd) {
        int[] topIndices = topKIndices(logits, topK);
        float[] probs = stableSoftmax(logits, topIndices, temperature);
        return categoricalSample(topIndices, probs, rnd);
    }

    private static int[] topKIndices(float[] logits, int topK) {
        int k = (topK == 0) ? logits.length : Math.min(topK, logits.length);
        if (k == logits.length) return indices(logits.length);
        int[] heap = new int[k];
        for (int i = 0; i < logits.length; i++) {
            offerTopIndex(heap, Math.min(i, k), i, logits);
        }
        sortDescending(heap, logits);
        return heap;
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
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
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

    private static void validateArgs(int maxNewTokens, float temperature, int topK) {
        if (maxNewTokens < 0) throw new IllegalArgumentException("maxNewTokens must be >= 0");
        if (!Float.isFinite(temperature) || temperature <= 0.0f)
            throw new IllegalArgumentException("temperature must be finite and > 0");
        if (topK < 0)         throw new IllegalArgumentException("topK must be >= 0");
    }

    private static void requirePrompt(int[] promptIds, int maxNewTokens) {
        if (promptIds.length == 0 && maxNewTokens > 0) {
            throw new IllegalArgumentException("prompt must encode to at least one token");
        }
    }

    private static int[] indices(int length) {
        int[] indices = new int[length];
        for (int i = 0; i < length; i++) {
            indices[i] = i;
        }
        return indices;
    }

    private static void offerTopIndex(int[] heap, int size, int index, float[] logits) {
        if (size < heap.length) {
            heap[size] = index;
            siftUp(heap, size, logits);
        } else if (isBetter(index, heap[0], logits)) {
            heap[0] = index;
            siftDown(heap, logits);
        }
    }

    private static void siftUp(int[] heap, int child, float[] logits) {
        while (child > 0) {
            int parent = (child - 1) / 2;
            if (!isWorse(heap[child], heap[parent], logits)) return;
            swap(heap, child, parent);
            child = parent;
        }
    }

    private static void siftDown(int[] heap, float[] logits) {
        int parent = 0;
        while (parent * 2 + 1 < heap.length) {
            int child = worseChild(heap, parent, logits);
            if (!isWorse(heap[child], heap[parent], logits)) return;
            swap(heap, parent, child);
            parent = child;
        }
    }

    private static int worseChild(int[] heap, int parent, float[] logits) {
        int left = parent * 2 + 1;
        int right = left + 1;
        if (right == heap.length || isWorse(heap[left], heap[right], logits)) return left;
        return right;
    }

    private static boolean isBetter(int left, int right, float[] logits) {
        int compared = Float.compare(logits[left], logits[right]);
        return compared > 0 || compared == 0 && left < right;
    }

    private static boolean isWorse(int left, int right, float[] logits) {
        int compared = Float.compare(logits[left], logits[right]);
        return compared < 0 || compared == 0 && left > right;
    }

    private static void sortDescending(int[] indices, float[] logits) {
        for (int i = 1; i < indices.length; i++) {
            insert(indices, i, logits);
        }
    }

    private static void insert(int[] indices, int position, float[] logits) {
        int value = indices[position];
        while (position > 0 && isBetter(value, indices[position - 1], logits)) {
            indices[position] = indices[position - 1];
            position--;
        }
        indices[position] = value;
    }

    private static void swap(int[] values, int left, int right) {
        int value = values[left];
        values[left] = values[right];
        values[right] = value;
    }

    private static int[] context(int[] ids, int length, int maxSeqLen) {
        int start = Math.max(0, length - maxSeqLen);
        return Arrays.copyOfRange(ids, start, length);
    }
}
