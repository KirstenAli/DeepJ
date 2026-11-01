package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.transformer.ModelLoader;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrism;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;

import java.io.IOException;
import java.nio.file.Path;

final class PrismCheckpoint {

    private PrismCheckpoint() {}

    static Loaded load(Path checkpoint) throws IOException {
        if (!(ModelLoader.load(checkpoint) instanceof DeepJPrism model)) {
            throw new IOException("Checkpoint is not a DeepJ Prism model");
        }
        Path tokenizerFile = checkpoint.toAbsolutePath().getParent().resolve("tokenizer.bpe");
        BPETokenizer tokenizer = new BPETokenizer(BPEModelIO.load(tokenizerFile));
        if (model.config().vocabSize() != tokenizer.vocabSize()) {
            throw new IOException("Tokenizer vocabulary does not match model");
        }
        return new Loaded(tokenizer, model.config(), model);
    }

    record Loaded(BPETokenizer tokenizer, DeepJPrismConfig config, DeepJPrism model) {}
}
