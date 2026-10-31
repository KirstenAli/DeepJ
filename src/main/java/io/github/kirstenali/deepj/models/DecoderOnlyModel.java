package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.layers.Projection;
import io.github.kirstenali.deepj.inference.InferenceContext;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.persistence.Persistable;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.persistence.ModelSettings;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.transformer.TransformerStack;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;
import io.github.kirstenali.deepj.layers.transformer.norm.NormLayer;

import java.util.ArrayList;
import java.util.List;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Map;

public abstract class DecoderOnlyModel implements CausalLM, Persistable {

    protected final Embedding        tokEmb;
    protected final TransformerStack stack;
    protected final NormLayer        normF;
    protected final Projection       lmHead;
    private PromptFormat promptFormat = PromptFormat.PLAIN;
    private Map<String, String> metadata = Map.of();

    protected DecoderOnlyModel(Embedding tokEmb, TransformerStack stack, NormLayer normF, Projection lmHead) {
        this.tokEmb = tokEmb;
        this.stack  = stack;
        this.normF  = normF;
        this.lmHead = lmHead;
    }

    public abstract TransformerConfig config();

    public final PromptFormat promptFormat() {
        return promptFormat;
    }

    public final void setPromptFormat(PromptFormat format) {
        promptFormat = Objects.requireNonNull(format, "format");
    }

    public final Map<String, String> metadata() {
        return metadata;
    }

    public final void setMetadata(Map<String, String> values) {
        metadata = Map.copyOf(values);
    }

    @Override
    public final void save(Path path) throws IOException {
        ModelSerializer.save(parameters(), path, new ModelSettings(config(), promptFormat, metadata));
    }

    @Override
    public final void load(Path path) throws IOException {
        ModelSettings settings = ModelSerializer.load(parameters(), path, config());
        if (settings != null) {
            promptFormat = settings.promptFormat();
            metadata = settings.metadata();
        }
    }

    protected Tensor embed(int[] inputIds) {
        return tokEmb.forward(inputIds);
    }

    protected Tensor embed(int[] inputIds, int offset) { return embed(inputIds); }

    protected Tensor embedBatch(int[][] inputIds) { return tokEmb.forwardBatch(inputIds); }

    protected boolean supportsCachedGeneration() { return false; }

    protected boolean supportsRollingGeneration() { return false; }

    @Override
    public GenerationSession openGeneration() {
        return openGeneration(ContextWindowMode.REBUILD);
    }

    @Override
    public GenerationSession openGeneration(ContextWindowMode mode) {
        Objects.requireNonNull(mode, "mode");
        if (mode == ContextWindowMode.ROLLING && !supportsRollingGeneration()) {
            throw new UnsupportedOperationException("This model does not support rolling cached generation");
        }
        return supportsCachedGeneration() && stack.supportsIncrementalInference()
                ? new CachedGenerationSession(this, mode) : uncachedGeneration(mode);
    }

    private GenerationSession uncachedGeneration(ContextWindowMode mode) {
        if (mode == ContextWindowMode.ROLLING) {
            throw new UnsupportedOperationException("The decoder stack does not support rolling cached generation");
        }
        return new UncachedGenerationSession(this);
    }

    final Tensor forwardCached(int[] inputIds, InferenceContext context) {
        try (var scope = TensorScope.open()) {
            return scope.escape(cachedLogits(inputIds, context));
        }
    }

    private Tensor cachedLogits(int[] inputIds, InferenceContext context) {
        int width = config().dModel();
        Tensor input = embed(inputIds, context.position()).reshape(1, inputIds.length, width);
        Tensor hidden = stack.forward(input, context).reshape(inputIds.length, width);
        Tensor last = Tensor.sliceRows(hidden, new int[]{inputIds.length - 1}, width).reshape(1, 1, width);
        return lmHead.forward(normF.forward(last).reshape(1, width));
    }

    protected List<Parameter> embeddingParameters() {
        return new ArrayList<>(tokEmb.parameters());
    }

    @Override
    public Tensor forward(int[] inputIds) {
        requireLength(inputIds.length);
        try (var scope = TensorScope.open()) {
            return scope.escape(forwardEmbedded(embed(inputIds)));
        }
    }

    @Override public final boolean supportsBatchedForward() { return true; }

    @Override
    public Tensor forwardBatch(int[][] inputIds) {
        if (inputIds == null || inputIds.length == 0 || inputIds[0] == null) {
            throw new IllegalArgumentException("A model batch must contain nonempty token sequences");
        }
        requireLength(inputIds[0].length);
        try (var scope = TensorScope.open()) {
            return scope.escape(forwardEmbedded(embedBatch(inputIds)));
        }
    }

    private Tensor forwardEmbedded(Tensor input) {
        return lmHead.forward(normF.forward(stack.forward(input)));
    }

    private void requireLength(int length) {
        if (length < 1 || length > config().maxSeqLen()) {
            throw new IllegalArgumentException("Sequence length exceeds the configured model context");
        }
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = embeddingParameters();
        ps.addAll(stack.parameters());
        ps.addAll(normF.parameters());
        ps.addAll(lmHead.parameters());
        return ps.stream().distinct().toList();
    }

    protected final void applyInitScale(float factor) {
        if (factor == 1.0f) return;
        try (var ignored = GradMode.noGrad()) {
            scaleInitialWeights(factor);
        }
    }

    private void scaleInitialWeights(float factor) {
        for (Parameter parameter : parameters()) {
            if (isRandomWeight(parameter.value)) parameter.value.multiplyScalarInPlace(factor);
        }
    }

    private static boolean isRandomWeight(Tensor tensor) {
        return !isFilledWith(tensor, 0.0f) && !isFilledWith(tensor, 1.0f);
    }

    private static boolean isFilledWith(Tensor tensor, float value) {
        float[] values = tensor.dtype() == DType.FLOAT32 ? tensor.rawData() : null;
        Tensor flat = values == null ? tensor.reshape(tensor.numel()) : null;
        for (int index = 0; index < tensor.numel(); index++) {
            if ((values == null ? flat.get(index) : values[index]) != value) return false;
        }
        return true;
    }
}
