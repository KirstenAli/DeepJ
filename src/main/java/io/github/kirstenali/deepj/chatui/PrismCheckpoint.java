package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class PrismCheckpoint {

    private static final String PROPERTIES_FILE = "training.properties";
    private static final String JSON_FILE = "config.json";

    private PrismCheckpoint() {}

    static Loaded load(Path checkpoint) throws IOException {
        Path directory = requireParent(checkpoint);
        Properties settings = settings(directory);
        BPETokenizer tokenizer = tokenizer(directory);
        DeepJPrismConfig config = config(settings, tokenizer.vocabSize());
        DeepJPrism model = new DeepJPrism(config, longValue(settings, "seed", 42L));
        model.load(checkpoint);
        return new Loaded(tokenizer, config, model);
    }

    private static Path requireParent(Path checkpoint) {
        Path directory = checkpoint.toAbsolutePath().getParent();
        if (directory == null) throw new IllegalArgumentException("Model must have a parent directory");
        return directory;
    }

    private static Properties settings(Path directory) throws IOException {
        Path properties = directory.resolve(PROPERTIES_FILE);
        if (Files.isRegularFile(properties)) return properties(properties);
        Path json = directory.resolve(JSON_FILE);
        if (Files.isRegularFile(json)) return json(json);
        throw new IOException("The model folder needs training.properties or config.json");
    }

    private static Properties properties(Path path) throws IOException {
        Properties result = new Properties();
        try (InputStream stream = Files.newInputStream(path)) {
            result.load(stream);
        }
        return result;
    }

    private static Properties json(Path path) throws IOException {
        String json = Files.readString(path);
        Properties result = new Properties();
        copyJson(json, result, "max_seq_len", "sequenceLength");
        copyJson(json, result, "d_model", "dModel");
        copyJson(json, result, "n_heads", "heads");
        copyJson(json, result, "n_layers", "layers");
        copyJson(json, result, "d_ff", "dFF");
        copyJson(json, result, "q_rank", "qRank");
        copyJson(json, result, "kv_rank", "kvRank");
        copyJson(json, result, "init_scale", "initScale");
        copyJson(json, result, "grad_clip_norm", "gradClipNorm");
        return result;
    }

    private static void copyJson(String json, Properties target, String source, String destination) {
        Matcher matcher = Pattern.compile("\\\"" + source + "\\\"\\s*:\\s*([^,}\\s]+)").matcher(json);
        if (!matcher.find()) throw new IllegalArgumentException("Missing config value: " + source);
        target.setProperty(destination, matcher.group(1).replace("\"", ""));
    }

    private static BPETokenizer tokenizer(Path directory) throws IOException {
        Path path = directory.resolve("tokenizer.bpe");
        if (!Files.isRegularFile(path)) throw new IOException("Missing tokenizer.bpe beside the model");
        return new BPETokenizer(BPEModelIO.load(path));
    }

    private static DeepJPrismConfig config(Properties values, int vocabularySize) {
        return new DeepJPrismConfig(vocabularySize, intValue(values, "sequenceLength"),
                intValue(values, "dModel"), intValue(values, "heads"),
                intValue(values, "layers"), intValue(values, "dFF"),
                intValue(values, "qRank"), intValue(values, "kvRank"),
                floatValue(values, "initScale", 0.2f),
                floatValue(values, "gradClipNorm", 1.0f));
    }

    private static int intValue(Properties values, String key) {
        return Integer.parseInt(values.getProperty(key));
    }

    private static long longValue(Properties values, String key, long fallback) {
        return Long.parseLong(values.getProperty(key, Long.toString(fallback)));
    }

    private static float floatValue(Properties values, String key, float fallback) {
        return Float.parseFloat(values.getProperty(key, Float.toString(fallback)));
    }

    record Loaded(BPETokenizer tokenizer, DeepJPrismConfig config, DeepJPrism model) {}
}
