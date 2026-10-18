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
        return formatModelCard(card, details, usage(format));
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
        return formatModelCard(card, details, usage(format));
    }

    private static String formatModelCard(ModelCard card, String details, String usage) {
        return CARD_TEMPLATE.formatted(card.license(), card.language(), card.name(), card.summary(),
                details, usage, card.trainingData(), card.limitations());
    }

    private static String usage(PromptFormat format) {
        String prompt = format == PromptFormat.INSTRUCTION_RESPONSE
                ? "What is a computer?" : "Once upon a time";
        return usageHeader() + """
                ```java
                DeepJChat chat = DeepJChat.load(Path.of("downloaded-model"));
                String answer = chat.generate("%s");
                ```
                """.formatted(prompt) + desktopUsage().stripTrailing();
    }

    private static String usageHeader() {
        return "Use [DeepJ 0.9.8-alpha](https://central.sonatype.com/artifact/"
                + "io.github.kirstenali/deepj/0.9.8-alpha) or newer:\n\n"
                + dependency() + "Download this repository's `model.dj` and `tokenizer.bpe` "
                + "into `downloaded-model`. Imports are omitted below.\n\n";
    }

    private static String dependency() {
        return """
                ```xml
                <dependency>
                    <groupId>io.github.kirstenali</groupId>
                    <artifactId>deepj</artifactId>
                    <version>0.9.8-alpha</version>
                </dependency>
                ```

                """;
    }

    private static String desktopUsage() {
        return """

                You can also use [DeepJ's desktop chat UI](https://github.com/KirstenAli/DeepJ#desktop-chat).
                Run it from the DeepJ repository with JDK 20 or newer:

                ```bash
                mvn org.openjfx:javafx-maven-plugin:0.0.8:run
                ```

                Then choose `model.dj`.
                Keep `tokenizer.bpe` in the same directory. The UI applies the checkpoint's saved
                prompt format automatically and streams the generated text.
                """ + generationSettings();
    }

    private static String generationSettings() {
        return """

                ### Generation settings

                | Setting | What it changes |
                |---|---|
                | Temperature | Controls the degree of randomness in token selection. |
                | Top-k | Samples from the top `k` token choices; `0` considers all. |
                | Max tokens | Limits new tokens, not words. |
                | Seed | Controls sampling randomness; changing it can change the response. |
                """ + "| Repetition penalty | Makes the model less likely to repeat itself. "
                + "`1.0` turns it off; try `1.1` if responses loop. |\n";
    }
}
