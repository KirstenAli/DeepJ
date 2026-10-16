package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbitConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.persistence.ModelSettings;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class LegacyModelSettings {

    private LegacyModelSettings() {}

    static ModelSettings read(Path checkpoint) throws IOException {
        Path folder = checkpoint.toAbsolutePath().getParent();
        Properties values = values(folder);
        try {
            TransformerConfig config = config(values);
            return new ModelSettings(config, promptFormat(folder, values));
        } catch (RuntimeException error) {
            throw new IOException("Invalid model sidecar configuration", error);
        }
    }

    private static PromptFormat promptFormat(Path folder, Properties values) {
        String fallback = Files.isRegularFile(folder.resolve("fine-tuning.properties"))
                ? "instruction_response" : "plain";
        return PromptFormat.valueOf(values.getProperty("promptFormat", fallback)
                .toUpperCase(Locale.ROOT));
    }

    private static Properties values(Path folder) throws IOException {
        Path training = folder.resolve("training.properties");
        if (Files.isRegularFile(training)) return properties(training);
        Path exported = folder.resolve("config.json");
        if (Files.isRegularFile(exported)) return json(exported);
        throw new IOException("Legacy model requires training.properties or config.json");
    }

    private static Properties properties(Path path) throws IOException {
        Properties values = new Properties();
        try (InputStream input = Files.newInputStream(path)) {
            values.load(input);
        }
        return values;
    }

    private static Properties json(Path path) throws IOException {
        String source = Files.readString(path);
        Properties values = new Properties();
        String[] fields = {"model_type:modelType", "prompt_format:promptFormat",
                "vocab_size:vocabSize",
                "max_seq_len:sequenceLength", "d_model:dModel", "n_heads:heads",
                "n_layers:layers", "d_ff:dFF", "q_rank:qRank", "kv_rank:kvRank",
                "init_scale:initScale", "grad_clip_norm:gradClipNorm"};
        for (String field : fields) {
            copyJson(source, values, field.split(":"));
        }
        return values;
    }

    private static void copyJson(String source, Properties values, String[] names) {
        Matcher match = Pattern.compile("\\\"" + names[0] + "\\\"\\s*:\\s*\\\"?([^,}\\s\\\"]+)")
                .matcher(source);
        if (match.find()) values.setProperty(names[1], match.group(1));
    }

    private static TransformerConfig config(Properties values) {
        String type = values.getProperty("modelType", inferredType(values));
        return switch (type) {
            case "deepj-origin" -> origin(values);
            case "deepj-orbit" -> orbit(values);
            case "deepj-prism" -> prism(values);
            default -> throw new IllegalArgumentException("Unsupported model type: " + type);
        };
    }

    private static String inferredType(Properties values) {
        return values.containsKey("qRank") ? "deepj-prism" : "deepj-origin";
    }

    private static DeepJOriginConfig origin(Properties values) {
        return new DeepJOriginConfig(integer(values, "vocabSize"), integer(values, "sequenceLength"),
                integer(values, "dModel"), integer(values, "heads"), integer(values, "layers"),
                integer(values, "dFF"), decimal(values, "initScale", 0.2f),
                decimal(values, "gradClipNorm", 1.0f));
    }

    private static DeepJOrbitConfig orbit(Properties values) {
        return new DeepJOrbitConfig(integer(values, "vocabSize"), integer(values, "sequenceLength"),
                integer(values, "dModel"), integer(values, "heads"), integer(values, "layers"),
                integer(values, "dFF"), decimal(values, "initScale", 0.2f),
                decimal(values, "gradClipNorm", 1.0f));
    }

    private static DeepJPrismConfig prism(Properties values) {
        return new DeepJPrismConfig(integer(values, "vocabSize"), integer(values, "sequenceLength"),
                integer(values, "dModel"), integer(values, "heads"), integer(values, "layers"),
                integer(values, "dFF"), integer(values, "qRank"), integer(values, "kvRank"),
                decimal(values, "initScale", 0.2f), decimal(values, "gradClipNorm", 1.0f));
    }

    private static int integer(Properties values, String key) {
        return Integer.parseInt(values.getProperty(key));
    }

    private static float decimal(Properties values, String key, float fallback) {
        return Float.parseFloat(values.getProperty(key, Float.toString(fallback)));
    }
}
