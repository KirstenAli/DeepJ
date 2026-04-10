package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.llama.LlamaConfig;
import io.github.kirstenali.deepj.models.llama.LlamaModel;
import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.Trainer;

import java.nio.file.Path;

/**
 * Example: tiny Llama training on a small text file using byte-level tokens.
 *
 * <p>Architecture differences from the GPT example:
 * <ul>
 *   <li>No learned positional embedding — RoPE is applied inside each attention block.</li>
 *   <li>RMSNorm instead of LayerNorm.</li>
 *   <li>SwiGLU feed-forward instead of GELU-FFN.</li>
 * </ul>
 *
 * Intended as a smoke test / reference, not for serious training.
 */
public final class TrainSmallLlama {

    public static void main(String[] args) throws Exception {

        // ── Backend ───────────────────────────────────────────────────────────
        MetalBackend metal = new MetalBackend();
        Tensor.setBackend(metal);

        // ── Data ──────────────────────────────────────────────────────────────
        Path corpus = Path.of("sample_data/llm_training_dataset_1227_examples.txt");

        Tokenizer tok = new ByteTokenizer();
        TextDataset ds = TextDataset.fromFile(corpus, tok, 256, 123);

        // ── Model ─────────────────────────────────────────────────────────────
        // dFF is computed by the Llama formula: round(8/3 * dModel) up to nearest 64
        int dModel = 512;
        LlamaConfig cfg = new LlamaConfig(
                tok.vocabSize(),
                256,                           // maxSeqLen
                dModel,                        // dModel
                4,                             // nHeads
                5,                             // nLayers
                LlamaConfig.defaultDFF(dModel) // dFF ≈ 1408 for dModel=512
        );

        LlamaModel model = new LlamaModel(cfg, 42);

        // ── Training ──────────────────────────────────────────────────────────
        Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-4f);

        // ── Checkpointing ─────────────────────────────────────────────────────
        Path checkpointDir = Path.of("checkpoints");

        Trainer.StepHook checkpointHook = (step, loss, ema) -> {
            if (step > 0 && step % 500 == 0) {
                model.save(checkpointDir.resolve("small-llama-" + step + ".bin"));
            }
        };

        trainer.train(
                10_000_000,    // maxSteps
                2,             // batchSize
                1,             // logEvery
                0.98f,         // emaBeta
                0.01f,         // targetEmaLoss
                25,            // releaseEverySteps – free orphaned GPU buffers every N steps
                checkpointHook // called after each step
        );

        Path finalModelPath = checkpointDir.resolve("small-llama-final.bin");
        model.save(finalModelPath);

        // ── Inference ─────────────────────────────────────────────────────────
        LlamaModel loadedModel = new LlamaModel(cfg, 42);
        loadedModel.load(finalModelPath);

        String prompt = "Bob Marley was ";
        String out = TextGenerator.generate(
                loadedModel,   // model
                tok,           // tokenizer
                cfg,           // config
                prompt,        // prompt text
                200,           // maxNewTokens
                0.1f,          // temperature
                20,            // topK
                1234L          // seed
        );

        System.out.println("\n=== Generated ===");
        System.out.println(out);
    }
}

