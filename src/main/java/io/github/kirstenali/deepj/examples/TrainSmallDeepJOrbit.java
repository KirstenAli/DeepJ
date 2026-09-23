package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.orbit.DeepJOrbitConfig;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbit;
import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.nio.file.Path;

public final class TrainSmallDeepJOrbit {

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        Tokenizer tok = new ByteTokenizer();
        TextDataset ds = TrainingExampleSupport.dataset(tok);
        int dModel = 512;
        DeepJOrbitConfig cfg = new DeepJOrbitConfig(tok.vocabSize(), 256, dModel, 4, 5,
                DeepJOrbitConfig.defaultDFF(dModel));
        DeepJOrbit model = new DeepJOrbit(cfg, 42);
        Path finalModelPath = TrainingExampleSupport.trainAndSave(model, model, ds, "small-deepj-orbit");
        DeepJOrbit loadedModel = new DeepJOrbit(cfg, 42);
        loadedModel.load(finalModelPath);
        TrainingExampleSupport.generate(loadedModel, tok, cfg, "Bob Marley was ");
    }
}
