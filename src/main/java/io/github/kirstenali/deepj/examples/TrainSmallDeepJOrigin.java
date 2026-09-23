package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.nio.file.Path;

public final class TrainSmallDeepJOrigin {

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        Tokenizer tok = new ByteTokenizer();
        TextDataset ds = TrainingExampleSupport.dataset(tok);
        DeepJOriginConfig cfg = new DeepJOriginConfig(tok.vocabSize(), 256, 512, 4, 5, 1024);
        DeepJOrigin model = new DeepJOrigin(cfg, 42);
        Path finalModelPath = TrainingExampleSupport.trainAndSave(model, model, ds, "small-deepj-origin");
        DeepJOrigin loadedModel = new DeepJOrigin(cfg, 42);
        loadedModel.load(finalModelPath);
        TrainingExampleSupport.generate(loadedModel, tok, cfg, "Bob Marley was ");
    }
}
