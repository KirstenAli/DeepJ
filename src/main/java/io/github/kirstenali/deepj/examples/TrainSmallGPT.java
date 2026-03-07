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
        Path corpus = Path.of("sample_data/AllCombined.txt");

        Tokenizer tok = new ByteTokenizer();
        TextDataset ds = TextDataset.fromFile(corpus, tok, 512, 123);

        GPTConfig cfg = new GPTConfig(
                ByteTokenizer.VOCAB_SIZE,
                512,     // maxSeqLen
                512,    // dModel
                8,      // nHeads
                8,      // nLayers
                2048 // dFF
        );

        GPTModel model = new GPTModel(cfg, 42);

        Trainer trainer = CausalLMTraining.trainer(model, ds, 3e-4);

        Path checkpointDir = Path.of("checkpoints");

        Trainer.StepHook checkpointHook = (step, loss, ema) -> {
            if (step > 0 && step % 500 == 0) {
                model.save(checkpointDir.resolve("small-gpt-" + step + ".bin"));
            }
        };

        // Train until target EMA loss or max steps.
        trainer.train(
                10_000_000,
                16,
                50,
                0.98,
                0.099,
                checkpointHook
        );

        Path finalModelPath = checkpointDir.resolve("small-gpt-final.bin");
        model.save(finalModelPath);

        GPTModel loadedModel = new GPTModel(cfg, 42);
        loadedModel.load(finalModelPath);

        // Generate a continuation.
        String prompt = "Mara wrote down the rhythm, ";
        String out = TextGenerator.generate(
                loadedModel,
                tok,
                cfg,
                prompt,
                200,
                0.1,
                20,
                1234L
        );

        System.out.println("\n=== Generated ===");
        System.out.println(out);
    }
}
