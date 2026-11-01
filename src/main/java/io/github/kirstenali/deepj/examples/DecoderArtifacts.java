package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.nn.transformer.DecoderOnlyModel;
import io.github.kirstenali.deepj.nn.transformer.ModelLoader;
import io.github.kirstenali.deepj.nn.transformer.TransformerConfig;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;

import java.io.IOException;
import java.nio.file.Path;

final class DecoderArtifacts {

    private DecoderArtifacts() {}

    static Loaded load(Path directory, Path checkpoint) throws IOException {
        var tokenizer = new BPETokenizer(BPEModelIO.load(directory.resolve("tokenizer.bpe")));
        DecoderOnlyModel model = ModelLoader.load(checkpoint);
        if (tokenizer.vocabSize() != model.config().vocabSize()) {
            throw new IOException("Tokenizer vocabulary does not match model");
        }
        return new Loaded(model, tokenizer);
    }

    record Loaded(DecoderOnlyModel model, BPETokenizer tokenizer) {
        TransformerConfig config() {
            return model.config();
        }
    }
}
