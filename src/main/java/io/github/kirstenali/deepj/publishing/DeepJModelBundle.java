package io.github.kirstenali.deepj.publishing;

import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.PromptFormat;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.persistence.ModelSettings;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModel;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Objects;
import java.util.HashMap;
import java.util.Map;

public final class DeepJModelBundle {

    public static final String MODEL_FILE = "model.dj";
    public static final String TOKENIZER_FILE = "tokenizer.bpe";
    public static final String CONFIG_FILE = "config.json";
    public static final String MODEL_CARD_FILE = "README.md";
    private static final String ORIGIN_CONFIG_TEMPLATE = """
            {
              "library_name": "deepj",
              "model_type": "deepj-origin",
              "prompt_format": "%s",
              "checkpoint_format_version": %d,
              "tokenizer_format_version": %d,
              "vocab_size": %d,
              "max_seq_len": %d,
              "d_model": %d,
              "n_heads": %d,
              "n_layers": %d,
              "d_ff": %d,
              "init_scale": %s,
              "grad_clip_norm": %s
            }
            """;
    private static final String PRISM_CONFIG_TEMPLATE = """
            {
              "library_name": "deepj",
              "model_type": "deepj-prism",
              "prompt_format": "%s",
              "checkpoint_format_version": %d,
              "tokenizer_format_version": %d,
              "vocab_size": %d,
              "max_seq_len": %d,
              "d_model": %d,
              "n_heads": %d,
              "n_layers": %d,
              "d_ff": %d,
              "q_rank": %d,
              "kv_rank": %d,
              "init_scale": %s,
              "grad_clip_norm": %s
            }
            """;
    private static final String CARD_TEMPLATE = """
            ---
            library_name: deepj
            license: %s
            language:
            - %s
            pipeline_tag: text-generation
            tags:
            - deepj
            ---

            # %s

            %s

            ## Model details

            %s
            The `model.dj` checkpoint and `tokenizer.bpe` files use DeepJ's versioned binary formats.
            They are not directly loadable by the Python Transformers library.

            ## Usage

            %s

            ## Training data

            %s

            ## Limitations

            %s
            """;

    private DeepJModelBundle() {}

    public static Path export(Path directory, DeepJOrigin model, BPEModel tokenizer,
                              ModelCard card) throws IOException {
        Objects.requireNonNull(model, "model");
        DeepJOriginConfig config = model.config();
        validate(directory, tokenizer, card, config.vocabSize());
        return writeBundle(directory, model, tokenizer,
                configJson(config, model.promptFormat()), modelCard(card, config, model.promptFormat()));
    }

    public static Path export(Path directory, DeepJPrism model, BPEModel tokenizer,
                              ModelCard card) throws IOException {
        Objects.requireNonNull(model, "model");
        DeepJPrismConfig config = model.config();
        validate(directory, tokenizer, card, config.vocabSize());
        return writeBundle(directory, model, tokenizer,
                configJson(config, model.promptFormat()), modelCard(card, config, model.promptFormat()));
    }

    private static Path writeBundle(Path directory, DecoderOnlyModel model, BPEModel tokenizer,
                                    String configJson, String modelCard) throws IOException {
        Files.createDirectories(directory);
        ModelSerializer.save(model.parameters(), directory.resolve(MODEL_FILE),
                new ModelSettings(model.config(), model.promptFormat(), publicMetadata(model)));
        BPEModelIO.save(directory.resolve(TOKENIZER_FILE), tokenizer);
        writeText(directory.resolve(CONFIG_FILE), configJson);
        writeText(directory.resolve(MODEL_CARD_FILE), modelCard);
        return directory;
    }

    private static Map<String, String> publicMetadata(DecoderOnlyModel model) {
        Map<String, String> values = new HashMap<>(model.metadata());
        values.keySet().removeIf(DeepJModelBundle::isLocalPathKey);
        return values;
    }

    private static boolean isLocalPathKey(String key) {
        return key.equals("corpus") || key.equals("output") || key.equals("initialModel")
                || key.equals("fineTuning.base") || key.equals("fineTuning.initialModel")
                || key.endsWith(".path");
    }

    private static void validate(Path directory, BPEModel tokenizer, ModelCard card,
                                 int expectedVocabSize) {
        Objects.requireNonNull(directory, "directory");
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(card, "card");
        if (expectedVocabSize != tokenizer.vocabSize()) {
            throw new IllegalArgumentException("Config and tokenizer vocabulary sizes differ");
        }
    }

    private static void writeText(Path path, String content) throws IOException {
        Files.writeString(path, content, StandardCharsets.UTF_8);
    }

    private static String configJson(DeepJOriginConfig config, PromptFormat format) {
        return ORIGIN_CONFIG_TEMPLATE.formatted(format.name().toLowerCase(Locale.ROOT),
                ModelSerializer.CURRENT_FORMAT_VERSION,
                BPEModel.CURRENT_FORMAT_VERSION,
                config.vocabSize(), config.maxSeqLen(), config.dModel(), config.nHeads(),
                config.nLayers(), config.dFF(), config.initScale(), config.gradClipNorm());
    }

    private static String configJson(DeepJPrismConfig config, PromptFormat format) {
        return PRISM_CONFIG_TEMPLATE.formatted(format.name().toLowerCase(Locale.ROOT),
                ModelSerializer.CURRENT_FORMAT_VERSION,
                BPEModel.CURRENT_FORMAT_VERSION, config.vocabSize(), config.maxSeqLen(),
                config.dModel(), config.nHeads(), config.nLayers(), config.dFF(),
                config.qRank(), config.kvRank(), config.initScale(), config.gradClipNorm());
    }

    private static String modelCard(ModelCard card, DeepJOriginConfig config, PromptFormat format) {
        String details = "This DeepJ Origin model has %d layers, a hidden size of %d, "
                .formatted(config.nLayers(), config.dModel())
                + "%d attention heads and a %,d-token vocabulary."
                .formatted(config.nHeads(), config.vocabSize());
        return formatModelCard(card, details, usage("DeepJOrigin", format));
    }

    private static String modelCard(ModelCard card, DeepJPrismConfig config, PromptFormat format) {
        String details = "This compact language model was created with "
                + "[DeepJ](https://github.com/KirstenAli/DeepJ) and uses the DeepJ Prism "
                + "Transformer architecture with %d layers, a hidden size of %d, "
                .formatted(config.nLayers(), config.dModel())
                + "%d attention heads, Q rank %d, KV rank %d and a %,d-token vocabulary."
                .formatted(config.nHeads(), config.qRank(), config.kvRank(), config.vocabSize())
                + "\n\nDeepJ Prism uses low-rank Q/KV attention with rotary embeddings, "
                + "RMSNorm and SwiGLU. It recalculates the full context for every generated token.";
        return formatModelCard(card, details, usage("DeepJPrism", format));
    }

    private static String formatModelCard(ModelCard card, String details, String usage) {
        return CARD_TEMPLATE.formatted(card.license(), card.language(), card.name(), card.summary(),
                details, usage, card.trainingData(), card.limitations());
    }

    private static String usage(String modelType, PromptFormat format) {
        String prompt = format == PromptFormat.INSTRUCTION_RESPONSE
                ? "What is a computer?" : "Once upon a time";
        return usageHeader() + """
                ```java
                Path directory = Path.of("downloaded-model");
                BPEModel bpe = BPEModelIO.load(directory.resolve("tokenizer.bpe"));
                BPETokenizer tokenizer = new BPETokenizer(bpe);
                %s model = (%s) ModelLoader.load(directory.resolve("model.dj"));
                String prompt = model.promptFormat().apply("%s");
                String text = TextGenerator.generate(model, tokenizer, model.config(),
                        prompt, 80, 0.8f, 40, 2026L);
                ```
                """.formatted(modelType, modelType, prompt);
    }

    private static String usageHeader() {
        return """
                Use a DeepJ release that supports checkpoint format %d, or build the current source.

                Download this repository's `model.dj` and `tokenizer.bpe` into `downloaded-model`.
                Imports are omitted below.

                """.formatted(ModelSerializer.CURRENT_FORMAT_VERSION);
    }
}
