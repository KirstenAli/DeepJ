package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.inference.InferenceContext;
import io.github.kirstenali.deepj.layers.Projection;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.layers.transformer.qwen3.Qwen3Block;
import io.github.kirstenali.deepj.layers.transformer.qwen3.Qwen3RotaryTables;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.GenerationSession;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.PackedQ8;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.tensor.TensorEncoding;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Inference-only Qwen3 weights, kept in Q8_0 blocks on the selected backend. */
public final class Qwen3Q8 implements AutoCloseable {
    private final Qwen3Config config;
    private final TensorBackend backend = Tensor.backend();
    private final PackedQ8 embedding, output;
    private final List<Qwen3Block> blocks;
    private final RMSNorm1D norm;
    private final List<Parameter> norms;
    private final Map<String, PackedQ8> weights;
    private boolean closed;

    Qwen3Q8(Qwen3Config config, Map<String, PackedQ8> weights) {
        this.config = config;
        this.weights = Map.copyOf(weights);
        embedding = weights.get("model.embed_tokens.weight");
        output = weights.getOrDefault("lm_head.weight", embedding);
        ParameterFactory factory = ParameterFactory.float32();
        blocks = blocks(factory);
        norm = new RMSNorm1D(config.dModel(), factory.scope("model.norm"));
        norms = norms();
    }

    private List<Qwen3Block> blocks(ParameterFactory factory) {
        List<Qwen3Block> result = new ArrayList<>();
        for (int layer = 0; layer < config.nLayers(); layer++) {
            String prefix = "model.layers." + layer;
            result.add(new Qwen3Block(config, factory.scope(prefix),
                    name -> new PackedProjection(weights.get(prefix + "." + name + ".weight"))));
        }
        return List.copyOf(result);
    }

    private List<Parameter> norms() {
        List<Parameter> result = new ArrayList<>();
        blocks.forEach(block -> result.addAll(block.parameters()));
        result.addAll(norm.parameters());
        result.forEach(parameter -> parameter.setGradient(null));
        return List.copyOf(result);
    }

    List<Parameter> normalizationWeights() { return norms; }
    public Qwen3Config config() { return config; }

    public GenerationSession openGeneration(ContextWindowMode mode) {
        requireOpen();
        return new Qwen3Session(config, norms, this::cached, this::requireTokens, Objects.requireNonNull(mode));
    }

    private Tensor cached(int[] tokens, InferenceContext cache) {
        try (var scope = TensorScope.open(); var tables = Qwen3RotaryTables.open()) {
            Tensor hidden = embedding.gather(tokens);
            for (Qwen3Block block : blocks) {
                hidden = block.forward(hidden, cache);
            }
            Tensor last = hidden.select(-2, tokens.length - 1).reshape(1, config.dModel());
            return scope.escape(output.project(norm.forward(last)));
        }
    }

    private void requireTokens(int[] tokens) {
        requireOpen();
        if (tokens == null || tokens.length < 1 || tokens.length > config.maxSeqLen()) {
            throw new IllegalArgumentException("Token context is empty or exceeds the configured length");
        }
    }

    private void requireOpen() {
        if (closed || backend != Tensor.backend()) throw new IllegalStateException("Q8_0 model is unavailable");
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        weights.values().forEach(PackedQ8::close);
        norms.forEach(parameter -> TensorEncoding.releaseOwnedWeight(parameter.value));
    }

    private record PackedProjection(PackedQ8 weight) implements Projection {
        @Override public Tensor forward(Tensor input) { return weight.project(input); }
        @Override public List<Parameter> parameters() { return List.of(); }
    }
}
