package io.github.kirstenali.deepj.publishing;

import io.github.kirstenali.deepj.models.deepseek.DeepSeekConfig;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.persistence.Persistable;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModel;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

public final class DeepJModelBundle {

    private static final String DEEPJ_VERSION = "0.7.0-alpha";
    public static final String MODEL_FILE = "model.dj";
    public static final String TOKENIZER_FILE = "tokenizer.bpe";
    public static final String CONFIG_FILE = "config.json";
    public static final String MODEL_CARD_FILE = "README.md";
    private static final String GPT_CONFIG_TEMPLATE = """
            {
              "library_name": "deepj",
              "model_type": "deepj-gpt",
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
    private static final String DEEPSEEK_CONFIG_TEMPLATE = """
            {
              "library_name": "deepj",
              "model_type": "deepj-deepseek-style",
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

    public static Path export(Path directory, GPTModel model, BPEModel tokenizer,
                              ModelCard card) throws IOException {
        Objects.requireNonNull(model, "model");
        GPTConfig config = model.config();
        validate(directory, tokenizer, card, config.vocabSize());
        return writeBundle(directory, model, tokenizer, configJson(config), modelCard(card, config));
    }

    public static Path export(Path directory, DeepSeekModel model, BPEModel tokenizer,
                              ModelCard card) throws IOException {
        Objects.requireNonNull(model, "model");
        DeepSeekConfig config = model.config();
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

    private static String configJson(GPTConfig config) {
        return GPT_CONFIG_TEMPLATE.formatted(ModelSerializer.CURRENT_FORMAT_VERSION, BPEModel.CURRENT_FORMAT_VERSION,
                config.vocabSize(), config.maxSeqLen(), config.dModel(), config.nHeads(),
                config.nLayers(), config.dFF(), config.initScale(), config.gradClipNorm());
    }

    private static String configJson(DeepSeekConfig config) {
        return DEEPSEEK_CONFIG_TEMPLATE.formatted(ModelSerializer.CURRENT_FORMAT_VERSION,
                BPEModel.CURRENT_FORMAT_VERSION, config.vocabSize(), config.maxSeqLen(),
                config.dModel(), config.nHeads(), config.nLayers(), config.dFF(),
                config.qRank(), config.kvRank(), config.initScale(), config.gradClipNorm());
    }

    private static String modelCard(ModelCard card, GPTConfig config) {
        String details = "This is a DeepJ GPT model with %d layers, width %d, %d attention heads, "
                .formatted(config.nLayers(), config.dModel(), config.nHeads())
                + "and a %d-token vocabulary.".formatted(config.vocabSize());
        return formatModelCard(card, details, usage(config));
    }

    private static String modelCard(ModelCard card, DeepSeekConfig config) {
        String details = "This compact language model was created with "
                + "[DeepJ](https://github.com/KirstenAli/DeepJ) and uses a DeepSeek-style "
                + "Transformer architecture with %d layers, a hidden size of %d, "
                .formatted(config.nLayers(), config.dModel())
                + "%d attention heads, Q rank %d, KV rank %d and a %,d-token vocabulary."
                .formatted(config.nHeads(), config.qRank(), config.kvRank(), config.vocabSize())
                + "\n\nIt is not an exact implementation of DeepSeek V2, V3 or R1 and does not "
                + "currently use an incremental KV cache.";
        return formatModelCard(card, details, usage(config));
    }

    private static String formatModelCard(ModelCard card, String details, String usage) {
        return CARD_TEMPLATE.formatted(card.license(), card.language(), card.name(), card.summary(),
                details, usage, card.trainingData(), card.limitations());
    }

    private static String usage(GPTConfig config) {
        return usageHeader() + """
                ```java
                Path directory = Path.of("downloaded-model");
                BPEModel bpe = BPEModelIO.load(directory.resolve("tokenizer.bpe"));
                BPETokenizer tokenizer = new BPETokenizer(bpe);
                GPTConfig config = new GPTConfig(%d, %d, %d, %d, %d, %d, %sf, %sf);
                GPTModel model = new GPTModel(config, 42L);
                model.load(directory.resolve("model.dj"));
                String text = TextGenerator.generate(model, tokenizer, config,
                        "Once upon a time", 80, 0.8f, 40, 2026L);
                ```
                """.formatted(config.vocabSize(), config.maxSeqLen(), config.dModel(),
                config.nHeads(), config.nLayers(), config.dFF(), config.initScale(),
                config.gradClipNorm());
    }

    private static String usage(DeepSeekConfig config) {
        return usageHeader() + """
                ```java
                Path directory = Path.of("downloaded-model");
                BPEModel bpe = BPEModelIO.load(directory.resolve("tokenizer.bpe"));
                BPETokenizer tokenizer = new BPETokenizer(bpe);
                DeepSeekConfig config = new DeepSeekConfig(
                        %d, %d, %d, %d, %d, %d, %d, %d, %sf, %sf);
                DeepSeekModel model = new DeepSeekModel(config, 42L);
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

                Download this repository's `model.dj` and `tokenizer.bpe` into `downloaded-model`. Imports are omitted below.

                """.formatted(DEEPJ_VERSION, DEEPJ_VERSION, DEEPJ_VERSION);
    }
}
