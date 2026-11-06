package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.inference.InferenceContext;
import io.github.kirstenali.deepj.layers.TiedEmbeddingProjection;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.layers.transformer.qwen3.Qwen3Block;
import io.github.kirstenali.deepj.layers.transformer.qwen3.Qwen3RotaryTables;
import io.github.kirstenali.deepj.nn.transformer.CausalLM;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.GenerationSession;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;
import io.github.kirstenali.deepj.training.Trainable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Supplier;

/** Qwen3 arithmetic uses DeepJ's existing tensor operations, autograd and selected backend. */
public final class Qwen3 implements CausalLM {
    private final Qwen3Config config;
    private final Embedding embedding;
    private final List<Qwen3Block> blocks;
    private final RMSNorm1D finalNorm;
    private final TiedEmbeddingProjection head;
    private final List<Parameter> parameters;

    public Qwen3(Qwen3Config config, long seed) { this(config, seed, ParameterFactory.float32(), true); }
    public Qwen3(Qwen3Config config, long seed, DType dtype) {
        this(config, seed, new ParameterFactory(dtype), true);
    }

    private Qwen3(Qwen3Config config, long seed, ParameterFactory factory, boolean initialize) {
        this.config = Objects.requireNonNull(config, "config");
        var random = new Random(seed);
        embedding = initialized(() -> new Embedding(config.vocabSize(), config.dModel(), random,
                factory.scope("model.embed_tokens")));
        blocks = blocks(config, random, factory);
        finalNorm = initialized(() -> new RMSNorm1D(config.dModel(), factory.scope("model.norm")));
        head = new TiedEmbeddingProjection(embedding.weight());
        parameters = inventory();
        if (initialize) initializeMatrices();
    }

    static Qwen3 forLoading(Qwen3Config config, ParameterFactory factory) {
        return new Qwen3(config, 0, factory.forLoading(), false);
    }

    private static List<Qwen3Block> blocks(Qwen3Config config, Random random, ParameterFactory factory) {
        List<Qwen3Block> result = new ArrayList<>();
        for (int index = 0; index < config.nLayers(); index++) {
            ParameterFactory parameters = factory.scope("model.layers." + index);
            result.add(initialized(() -> new Qwen3Block(config, random, parameters)));
        }
        return List.copyOf(result);
    }

    /** Inference does not retain unused zero gradients; existing autograd allocates them on backward. */
    private static <T extends Trainable> T initialized(Supplier<T> constructor) {
        try (var scope = TensorScope.open()) {
            T layer = constructor.get();
            for (Parameter parameter : layer.parameters()) {
                parameter.setGradient(null);
                scope.escape(parameter.value);
            }
            return layer;
        }
    }

    private List<Parameter> inventory() {
        List<Parameter> result = new ArrayList<>(embedding.parameters());
        blocks.forEach(block -> result.addAll(block.parameters()));
        result.addAll(finalNorm.parameters());
        return List.copyOf(result);
    }

    private void initializeMatrices() {
        try (var ignored = GradMode.noGrad()) {
            for (Parameter parameter : parameters) {
                if (!parameter.name().endsWith(".gamma")) parameter.value.multiplyScalarInPlace(config.initScale());
            }
        }
    }

    public Qwen3Config config() { return config; }
    @Override public List<Parameter> parameters() { return parameters; }
    @Override public float gradClipNorm() { return config.gradClipNorm(); }
    @Override public boolean supportsBatchedForward() { return true; }

    @Override public Tensor forward(int[] tokens) {
        requireLength(tokens);
        try (var scope = TensorScope.open()) { return scope.escape(logits(embedding.forward(tokens), null, false)); }
    }

    @Override public Tensor forwardBatch(int[][] tokens) {
        if (tokens == null || tokens.length == 0) throw new IllegalArgumentException("Empty token batch");
        requireLength(tokens[0]);
        try (var scope = TensorScope.open()) {
            return scope.escape(logits(embedding.forwardBatch(tokens), null, false));
        }
    }

    Tensor cached(int[] tokens, InferenceContext context) {
        requireLength(tokens);
        try (var scope = TensorScope.open()) {
            Tensor input = embedding.forward(tokens).reshape(1, tokens.length, config.dModel());
            return scope.escape(logits(input, context, true));
        }
    }

    private Tensor logits(Tensor hidden, InferenceContext context, boolean lastOnly) {
        try (var tables = Qwen3RotaryTables.open()) { return projected(hidden, context, lastOnly); }
    }

    private Tensor projected(Tensor hidden, InferenceContext context, boolean lastOnly) {
        for (Qwen3Block block : blocks) {
            hidden = context == null ? block.forward(hidden) : block.forward(hidden, context);
        }
        if (lastOnly) hidden = hidden.select(-2, hidden.size(-2) - 1).reshape(1, config.dModel());
        return head.forward(finalNorm.forward(hidden));
    }

    void requireLength(int[] tokens) {
        if (tokens == null || tokens.length < 1 || tokens.length > config.maxSeqLen()) {
            throw new IllegalArgumentException("Token context is empty or exceeds the configured length");
        }
    }

    @Override public GenerationSession openGeneration() { return openGeneration(ContextWindowMode.REBUILD); }
    @Override public GenerationSession openGeneration(ContextWindowMode mode) {
        return new Qwen3Session(this, Objects.requireNonNull(mode, "mode"));
    }
}
