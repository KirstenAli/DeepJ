package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

final class DeepJPrismTinyStoriesArtifacts {

    private DeepJPrismTinyStoriesArtifacts() {}

    static Loaded load(Path output, Path checkpoint) throws IOException {
        Properties properties = properties(output.resolve("training.properties"));
        BPETokenizer tokenizer = tokenizer(output.resolve("tokenizer.bpe"));
        DeepJPrismConfig config = config(properties, tokenizer.vocabSize());
        DeepJPrism model = new DeepJPrism(config, longValue(properties, "seed"));
        model.load(checkpoint);
        return new Loaded(properties, tokenizer, config, model);
    }

    private static Properties properties(Path path) throws IOException {
        Properties properties = new Properties();
        try (InputStream stream = Files.newInputStream(path)) {
            properties.load(stream);
        }
        return properties;
    }

    private static BPETokenizer tokenizer(Path path) throws IOException {
        return new BPETokenizer(BPEModelIO.load(path));
    }

    private static DeepJPrismConfig config(Properties properties, int vocabSize) {
        return new DeepJPrismConfig(vocabSize, intValue(properties, "sequenceLength"),
                intValue(properties, "dModel"), intValue(properties, "heads"),
                intValue(properties, "layers"), intValue(properties, "dFF"),
                intValue(properties, "qRank"), intValue(properties, "kvRank"),
                floatValue(properties, "initScale"), floatValue(properties, "gradClipNorm", 1.0f));
    }

    static int intValue(Properties properties, String key) {
        return Integer.parseInt(properties.getProperty(key));
    }

    static long longValue(Properties properties, String key) {
        return Long.parseLong(properties.getProperty(key));
    }

    private static float floatValue(Properties properties, String key) {
        return Float.parseFloat(properties.getProperty(key));
    }

    private static float floatValue(Properties properties, String key, float fallback) {
        return Float.parseFloat(properties.getProperty(key, Float.toString(fallback)));
    }

    record Loaded(Properties properties, BPETokenizer tokenizer,
                  DeepJPrismConfig config, DeepJPrism model) {}
}
