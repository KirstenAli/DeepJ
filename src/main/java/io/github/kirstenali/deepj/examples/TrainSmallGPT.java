package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.models.gpt.TextGenerator;
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
        Path corpus = Path.of("sample_data/llm_training_dataset_1227_examples.txt");

        Tokenizer tok = new ByteTokenizer();
        TextDataset ds = TextDataset.fromFile(corpus, tok, 128, 123);

        GPTConfig cfg = new GPTConfig(
                tok.vocabSize(),
                128,  // maxSeqLen
                256,             // dModel
                4,               // nHeads
                5,               // nLayers
                1024             // dFF
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
                0.01,
                checkpointHook
        );

        Path finalModelPath = checkpointDir.resolve("small-gpt-final.bin");
        model.save(finalModelPath);

        GPTModel loadedModel = new GPTModel(cfg, 42);
        loadedModel.load(finalModelPath);

        // Generate a continuation.
        String prompt = "Bob Marley was ";
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
