package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;

import java.nio.file.Path;

public final class TrainSmallGPTWithBPE {

    private static final int BPE_VOCAB_SIZE = 4_000;

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        Path tokenizerPath = TrainingExampleSupport.checkpointPath("small-gpt-bpe.tokenizer");
        BPETokenizer tok = trainTokenizer(tokenizerPath);
        TextDataset ds = TrainingExampleSupport.dataset(tok);
        GPTConfig cfg = new GPTConfig(tok.vocabSize(), 256, 512, 4, 5, 1024);
        GPTModel model = new GPTModel(cfg, 42);
        Path finalModelPath = TrainingExampleSupport.trainAndSave(model, model, ds, "small-gpt-bpe");
        BPETokenizer loadedTok = new BPETokenizer(BPEModelIO.load(tokenizerPath));
        GPTModel loadedModel = new GPTModel(cfg, 42);
        loadedModel.load(finalModelPath);
        TrainingExampleSupport.generate(loadedModel, loadedTok, cfg, "<BOS> Bob Marley was ");
    }

    private static BPETokenizer trainTokenizer(Path tokenizerPath) throws Exception {
        System.out.println("Training BPE tokenizer (vocab=" + BPE_VOCAB_SIZE + ") …");
        BPETokenizer tokenizer = new BPETrainer().trainTokenizerWithDefaultsFromFile(
                TrainingExampleSupport.CORPUS, BPE_VOCAB_SIZE);
        BPEModelIO.save(tokenizerPath, tokenizer.model());
        System.out.println("Tokenizer saved → " + tokenizerPath);
        return tokenizer;
    }
}
