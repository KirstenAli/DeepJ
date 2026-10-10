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

    private PrismCheckpoint() {}

    static Loaded load(Path checkpoint) throws IOException {
        Path folder = checkpoint.toAbsolutePath().getParent();
        Properties values = settings(folder);
        BPETokenizer tokenizer = new BPETokenizer(BPEModelIO.load(folder.resolve("tokenizer.bpe")));
        DeepJPrismConfig config = config(values, tokenizer.vocabSize());
        DeepJPrism model = new DeepJPrism(config, number(values, "seed", 42L));
        model.load(checkpoint);
        return new Loaded(tokenizer, config, model);
    }

    private static Properties settings(Path folder) throws IOException {
        Path path = folder.resolve("training.properties");
        if (Files.isRegularFile(path)) return properties(path);
        path = folder.resolve("config.json");
        if (Files.isRegularFile(path)) return json(path);
        throw new IOException("Model folder needs training.properties or config.json");
    }

    private static Properties properties(Path path) throws IOException {
        Properties values = new Properties();
        try (InputStream stream = Files.newInputStream(path)) { values.load(stream); }
        return values;
    }

    private static Properties json(Path path) throws IOException {
        String source = Files.readString(path);
        Properties values = new Properties();
        String[] fields = {"max_seq_len:sequenceLength", "d_model:dModel", "n_heads:heads",
                "n_layers:layers", "d_ff:dFF", "q_rank:qRank", "kv_rank:kvRank",
                "init_scale:initScale", "grad_clip_norm:gradClipNorm"};
        for (String field : fields) {
            copyJson(source, values, field.split(":"));
        }
        return values;
    }

    private static void copyJson(String source, Properties values, String[] names) {
        Matcher match = Pattern.compile("\\\"" + names[0] + "\\\"\\s*:\\s*([^,}\\s]+)").matcher(source);
        if (!match.find()) throw new IllegalArgumentException("Missing config value: " + names[0]);
        values.setProperty(names[1], match.group(1).replace("\"", ""));
    }

    private static DeepJPrismConfig config(Properties values, int vocab) {
        return new DeepJPrismConfig(vocab, integer(values, "sequenceLength"),
                integer(values, "dModel"), integer(values, "heads"), integer(values, "layers"),
                integer(values, "dFF"), integer(values, "qRank"), integer(values, "kvRank"),
                decimal(values, "initScale", 0.2f), decimal(values, "gradClipNorm", 1.0f));
    }

    private static int integer(Properties values, String key) {
        return Integer.parseInt(values.getProperty(key));
    }

    private static long number(Properties values, String key, long fallback) {
        return Long.parseLong(values.getProperty(key, Long.toString(fallback)));
    }

    private static float decimal(Properties values, String key, float fallback) {
        return Float.parseFloat(values.getProperty(key, Float.toString(fallback)));
    }

    record Loaded(BPETokenizer tokenizer, DeepJPrismConfig config, DeepJPrism model) {}
}
