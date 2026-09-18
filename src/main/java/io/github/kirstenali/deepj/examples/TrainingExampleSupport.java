package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.models.CausalLM;
import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.models.TransformerConfig;
import io.github.kirstenali.deepj.persistence.Persistable;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.Trainer;

import java.nio.file.Path;

final class TrainingExampleSupport {

    static final Path CORPUS = Path.of("sample_data/llm_training_dataset_1227_examples.txt");
    private static final Path CHECKPOINT_DIR = Path.of("checkpoints");

    private TrainingExampleSupport() {}

    static void configureBackend() {
        if (MetalBackend.isAvailable()) Tensor.setBackend(new MetalBackend());
    }

    static TextDataset dataset(Tokenizer tokenizer) throws Exception {
        return TextDataset.fromFile(CORPUS, tokenizer, 256, 123L);
    }

    static Path trainAndSave(CausalLM model, Persistable persistable, TextDataset dataset,
                             String checkpointName) throws Exception {
        Trainer trainer = CausalLMTraining.trainer(model, dataset, 1e-4f);
        trainer.train(10_000_000, 2, 1, 0.98f, 0.01f, 25,
                checkpointHook(persistable, checkpointName));
        Path finalPath = CHECKPOINT_DIR.resolve(checkpointName + "-final.bin");
        persistable.save(finalPath);
        return finalPath;
    }

    static void generate(CausalLM model, Tokenizer tokenizer, TransformerConfig config, String prompt) {
        String output = TextGenerator.generate(model::forward, config, tokenizer,
                prompt, 200, 0.1f, 20, 1234L);
        System.out.println("\n=== Generated ===");
        System.out.println(output);
    }

    static Path checkpointPath(String fileName) {
        return CHECKPOINT_DIR.resolve(fileName);
    }

    private static Trainer.StepHook checkpointHook(Persistable model, String name) {
        return (step, loss, ema) -> {
            if (step > 0 && step % 500 == 0) {
                model.save(CHECKPOINT_DIR.resolve(name + "-" + step + ".bin"));
            }
        };
    }
}
