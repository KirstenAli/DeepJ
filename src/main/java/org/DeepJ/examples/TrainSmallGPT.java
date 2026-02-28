package org.DeepJ.examples;

import org.DeepJ.ann.gpt.*;
import org.DeepJ.ann.tokenizer.ByteTokenizer;
import org.DeepJ.ann.training.CausalLMTraining;
import org.DeepJ.ann.training.Trainer;

import java.nio.file.Path;

/**
 * Example: tiny GPT training on a small text file using byte-level tokens.
 * Intended as a smoke test / reference, not for serious training.
 */
public final class TrainSmallGPT {

    public static void main(String[] args) throws Exception {
        Path corpus = Path.of("sample_data/sample_corpus.txt");

        ByteTokenizer tok = new ByteTokenizer();
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

        Trainer trainer = CausalLMTraining.trainer(model, ds, 3e-4);

        // Train until target EMA loss or max steps.
        trainer.train(
                10_000, // maxSteps
                16,     // batchSize
                50,     // logEvery
                0.98,   // emaBeta
                1.25    // targetEmaLoss (tune based on corpus size)
        );

        // Generate a continuation.
        String prompt = "In the workshop, the engineer wrote a note: ";
        String out = TextGenerator.generate(
                model,
                tok,
                cfg,
                prompt,
                200,    // maxNewTokens
                0.9,    // temperature
                50,     // topK (0 disables)
                1234L   // seed
        );

        System.out.println("\n=== Generated ===");
        System.out.println(out);
    }
}
