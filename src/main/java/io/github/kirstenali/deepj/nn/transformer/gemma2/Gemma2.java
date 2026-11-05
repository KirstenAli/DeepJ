package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.nn.transformer.CausalLM;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.GenerationSession;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.Arrays;
import java.util.List;

/** Gemma 2 language model reusing the same backbone as image prompt encoding. */
public final class Gemma2 implements CausalLM, AutoCloseable {
    private final Gemma2Config config;
    private final TensorWeights weights;
    private final Gemma2Backbone backbone;
    private final String prefix;

    public Gemma2(Gemma2Config config, TensorWeights weights) {
        this.config = java.util.Objects.requireNonNull(config);
        this.weights = java.util.Objects.requireNonNull(weights);
        for (String name : weights.descriptors().keySet()) { weights.tensor(name); }
        prefix = weights.contains("model.embed_tokens.weight") ? "model." : "";
        backbone = new Gemma2Backbone(config, name -> weights.tensor(prefix + name));
    }

    public Gemma2Config config() { return config; }
    @Override public List<Parameter> parameters() { return weights.parameters(); }
    @Override public float gradClipNorm() { return config.gradClipNorm(); }
    @Override public Tensor forward(int[] tokens) { return logits(tokens, null); }
    @Override public GenerationSession openGeneration() { return new Gemma2Session(this); }

    @Override public GenerationSession openGeneration(ContextWindowMode mode) {
        if (java.util.Objects.requireNonNull(mode) == ContextWindowMode.ROLLING) {
            throw new UnsupportedOperationException("Gemma 2 chat supports the standard context cache");
        }
        return openGeneration();
    }

    Tensor logits(int[] tokens, Gemma2Cache cache) {
        try (var scope = TensorScope.open()) {
            int[] valid = new int[tokens.length];
            Arrays.fill(valid, 1);
            Tensor hidden = backbone.forward(tokens, valid, cache);
            if (cache != null) hidden = hidden.slice(1, tokens.length - 1, tokens.length);
            String head = weights.contains("lm_head.weight") ? "lm_head.weight" : prefix + "embed_tokens.weight";
            Tensor logits = hidden.matmul(weights.tensor(head).transpose(-1, -2));
            return scope.escape(logits.divideScalar(config.logitSoftcap()).tanhActivation()
                    .multiplyScalar(config.logitSoftcap()).reshape(-1, config.vocabSize()));
        }
    }

    @Override public void close() { weights.close(); }
}
