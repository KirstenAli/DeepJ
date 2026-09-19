package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.llama.LlamaConfig;
import io.github.kirstenali.deepj.models.llama.LlamaModel;
import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.nio.file.Path;

public final class TrainSmallLlama {

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        Tokenizer tok = new ByteTokenizer();
        TextDataset ds = TrainingExampleSupport.dataset(tok);
        int dModel = 512;
        LlamaConfig cfg = new LlamaConfig(tok.vocabSize(), 256, dModel, 4, 5,
                LlamaConfig.defaultDFF(dModel));
        LlamaModel model = new LlamaModel(cfg, 42);
        Path finalModelPath = TrainingExampleSupport.trainAndSave(model, model, ds, "small-llama");
        LlamaModel loadedModel = new LlamaModel(cfg, 42);
        loadedModel.load(finalModelPath);
        TrainingExampleSupport.generate(loadedModel, tok, cfg, "Bob Marley was ");
    }
}
