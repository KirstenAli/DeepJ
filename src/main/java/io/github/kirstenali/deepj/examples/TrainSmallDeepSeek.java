package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.deepseek.DeepSeekConfig;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.nio.file.Path;

public final class TrainSmallDeepSeek {

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        Tokenizer tok = new ByteTokenizer();
        TextDataset ds = TrainingExampleSupport.dataset(tok);
        int dModel = 512;
        DeepSeekConfig cfg = new DeepSeekConfig(tok.vocabSize(), 256, dModel, 4, 5,
                1024, dModel / 2, dModel / 4);
        DeepSeekModel model = new DeepSeekModel(cfg, 42);
        Path finalModelPath = TrainingExampleSupport.trainAndSave(model, model, ds, "small-deepseek");
        DeepSeekModel loadedModel = new DeepSeekModel(cfg, 42);
        loadedModel.load(finalModelPath);
        TrainingExampleSupport.generate(loadedModel, tok, cfg, "Bob Marley was ");
    }
}
