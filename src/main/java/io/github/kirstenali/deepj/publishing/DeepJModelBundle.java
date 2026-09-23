package io.github.kirstenali.deepj.publishing;

import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.persistence.Persistable;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModel;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

public final class DeepJModelBundle {

    private static final String DEEPJ_VERSION = "0.8.0-alpha";
    public static final String MODEL_FILE = "model.dj";
    public static final String TOKENIZER_FILE = "tokenizer.bpe";
    public static final String CONFIG_FILE = "config.json";
    public static final String MODEL_CARD_FILE = "README.md";
    private static final String ORIGIN_CONFIG_TEMPLATE = """
            {
              "library_name": "deepj",
              "model_type": "deepj-origin",
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
        return writeBundle(directory, model, tokenizer, configJson(config), modelCard(card, config));
    }

    public static Path export(Path directory, DeepJPrism model, BPEModel tokenizer,
                              ModelCard card) throws IOException {
        Objects.requireNonNull(model, "model");
        DeepJPrismConfig config = model.config();
        validate(directory, tokenizer, card, config.vocabSize());
        return writeBundle(directory, model, tokenizer, configJson(config), modelCard(card, config));
    }

    private static Path writeBundle(Path directory, Persistable model, BPEModel tokenizer,
                                    String configJson, String modelCard) throws IOException {
        Files.createDirectories(directory);
        model.save(directory.resolve(MODEL_FILE));
        BPEModelIO.save(directory.resolve(TOKENIZER_FILE), tokenizer);
        writeText(directory.resolve(CONFIG_FILE), configJson);
        writeText(directory.resolve(MODEL_CARD_FILE), modelCard);
        return directory;
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

    private static String configJson(DeepJOriginConfig config) {
        return ORIGIN_CONFIG_TEMPLATE.formatted(ModelSerializer.CURRENT_FORMAT_VERSION,
                BPEModel.CURRENT_FORMAT_VERSION,
                config.vocabSize(), config.maxSeqLen(), config.dModel(), config.nHeads(),
                config.nLayers(), config.dFF(), config.initScale(), config.gradClipNorm());
    }

    private static String configJson(DeepJPrismConfig config) {
        return PRISM_CONFIG_TEMPLATE.formatted(ModelSerializer.CURRENT_FORMAT_VERSION,
                BPEModel.CURRENT_FORMAT_VERSION, config.vocabSize(), config.maxSeqLen(),
                config.dModel(), config.nHeads(), config.nLayers(), config.dFF(),
                config.qRank(), config.kvRank(), config.initScale(), config.gradClipNorm());
    }

    private static String modelCard(ModelCard card, DeepJOriginConfig config) {
        String details = "This DeepJ Origin model has %d layers, a hidden size of %d, "
                .formatted(config.nLayers(), config.dModel())
                + "%d attention heads and a %,d-token vocabulary."
                .formatted(config.nHeads(), config.vocabSize());
        return formatModelCard(card, details, usage(config));
    }

    private static String modelCard(ModelCard card, DeepJPrismConfig config) {
        String details = "This compact language model was created with "
                + "[DeepJ](https://github.com/KirstenAli/DeepJ) and uses the DeepJ Prism "
                + "Transformer architecture with %d layers, a hidden size of %d, "
                .formatted(config.nLayers(), config.dModel())
                + "%d attention heads, Q rank %d, KV rank %d and a %,d-token vocabulary."
                .formatted(config.nHeads(), config.qRank(), config.kvRank(), config.vocabSize())
                + "\n\nDeepJ Prism uses low-rank Q/KV attention with rotary embeddings, "
                + "RMSNorm and SwiGLU. It recalculates the full context for every generated token.";
        return formatModelCard(card, details, usage(config));
    }

    private static String formatModelCard(ModelCard card, String details, String usage) {
        return CARD_TEMPLATE.formatted(card.license(), card.language(), card.name(), card.summary(),
                details, usage, card.trainingData(), card.limitations());
    }

    private static String usage(DeepJOriginConfig config) {
        return usageHeader() + """
                ```java
                Path directory = Path.of("downloaded-model");
                BPEModel bpe = BPEModelIO.load(directory.resolve("tokenizer.bpe"));
                BPETokenizer tokenizer = new BPETokenizer(bpe);
                DeepJOriginConfig config = new DeepJOriginConfig(%d, %d, %d, %d, %d, %d, %sf, %sf);
                DeepJOrigin model = new DeepJOrigin(config, 42L);
                model.load(directory.resolve("model.dj"));
                String text = TextGenerator.generate(model, tokenizer, config,
                        "Once upon a time", 80, 0.8f, 40, 2026L);
                ```
                """.formatted(config.vocabSize(), config.maxSeqLen(), config.dModel(),
                config.nHeads(), config.nLayers(), config.dFF(), config.initScale(),
                config.gradClipNorm());
    }

    private static String usage(DeepJPrismConfig config) {
        return usageHeader() + """
                ```java
                Path directory = Path.of("downloaded-model");
                BPEModel bpe = BPEModelIO.load(directory.resolve("tokenizer.bpe"));
                BPETokenizer tokenizer = new BPETokenizer(bpe);
                DeepJPrismConfig config = new DeepJPrismConfig(
                        %d, %d, %d, %d, %d, %d, %d, %d, %sf, %sf);
                DeepJPrism model = new DeepJPrism(config, 42L);
                model.load(directory.resolve("model.dj"));
                String text = TextGenerator.generate(model, tokenizer, config,
                        "Once upon a time", 80, 0.8f, 40, 2026L);
                ```
                """.formatted(config.vocabSize(), config.maxSeqLen(), config.dModel(),
                config.nHeads(), config.nLayers(), config.dFF(), config.qRank(), config.kvRank(),
                config.initScale(), config.gradClipNorm());
    }

    private static String usageHeader() {
        return """
                Use [DeepJ %s from Maven Central](https://central.sonatype.com/artifact/io.github.kirstenali/deepj/%s), or a later format-compatible release.

                ```xml
                <dependency>
                    <groupId>io.github.kirstenali</groupId>
                    <artifactId>deepj</artifactId>
                    <version>%s</version>
                </dependency>
                ```

                Download this repository's `model.dj` and `tokenizer.bpe` into `downloaded-model`.
                Imports are omitted below.

                """.formatted(DEEPJ_VERSION, DEEPJ_VERSION, DEEPJ_VERSION);
    }
}
