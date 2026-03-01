package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.gpt.GPTConfig;
import io.github.kirstenali.deepj.gpt.GPTModel;
import io.github.kirstenali.deepj.gpt.TextDataset;
import io.github.kirstenali.deepj.gpt.TextGenerator;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.Trainer;

import java.nio.file.Path;

/**
 * Example: tiny GPT training on a small text file using byte-level tokens.
 * Intended as a smoke test / reference, not for serious training.
 */
public final class TrainSmallGPT {

    public static void main(String[] args) throws Exception {
        Path corpus = Path.of("sample_data/sample_corpus.txt");

        Tokenizer tok = new ByteTokenizer();
        TextDataset ds = TextDataset.fromFile(corpus, tok, 64, 123);

        GPTConfig cfg = new GPTConfig(
                ByteTokenizer.VOCAB_SIZE,
                64,     // maxSeqLen
                128,    // dModel
                4,      // nHeads
                2,      // nLayers
                4 * 128 // dFF
        );

        GPTModel model = new GPTModel(cfg, 42);

        Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-3);

        // Train until target EMA loss or max steps.
        trainer.train(
                10_000, // maxSteps
                16,     // batchSize
                50,     // logEvery
                0.98,   // emaBeta
                0.099    // targetEmaLoss (tune based on corpus size)
        );

        // Generate a continuation.
        String prompt = "Mara wrote down the rhythm, ";
        String out = TextGenerator.generate(
                model,
                tok,
                cfg,
                prompt,
                200,    // maxNewTokens
                0.1,    // temperature
                20,     // topK (0 disables)
                1234L   // seed
        );

        System.out.println("\n=== Generated ===");
        System.out.println(out);
    }
}
