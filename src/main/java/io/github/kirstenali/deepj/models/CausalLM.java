package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.Trainable;

/**
 * Common contract for decoder-only causal language models (GPT, Llama, DeepSeek, …).
 *
 * <p>Extends {@link Trainable} with the three additional methods that every
 * causal-LM training loop needs:
 * <ul>
 *   <li>{@link #forward} — token ids → logits {@code [seqLen × vocabSize]}</li>
 *   <li>{@link #backward} — back-propagate gradient of the logits</li>
 *   <li>{@link #gradClipNorm} — global gradient-clipping threshold</li>
 * </ul>
 */
public interface CausalLM extends Trainable {

    /** Maps token ids to logits {@code [seqLen × vocabSize]}. */
    Tensor forward(int[] inputIds);

    /** Back-propagates gradient of the logits through the model. */
    void backward(Tensor dLogits);

    /** Global gradient-clipping threshold (e.g. {@code 1.0}). */
    float gradClipNorm();
}

