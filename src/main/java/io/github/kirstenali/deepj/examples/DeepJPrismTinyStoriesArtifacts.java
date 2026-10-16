package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.ModelLoader;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

final class DeepJPrismTinyStoriesArtifacts {

    private DeepJPrismTinyStoriesArtifacts() {}

    static Loaded load(Path output, Path checkpoint) throws IOException {
        if (!(ModelLoader.load(checkpoint) instanceof DeepJPrism model)) {
            throw new IOException("Checkpoint is not a DeepJ Prism model");
        }
        BPETokenizer tokenizer = new BPETokenizer(BPEModelIO.load(output.resolve("tokenizer.bpe")));
        DeepJPrismConfig config = model.config();
        if (config.vocabSize() != tokenizer.vocabSize()) {
            throw new IOException("Tokenizer vocabulary does not match model");
        }
        return new Loaded(model.metadata(), tokenizer, config, model);
    }

    record Loaded(Map<String, String> metadata, BPETokenizer tokenizer,
                  DeepJPrismConfig config, DeepJPrism model) {}
}
