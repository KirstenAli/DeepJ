package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.deepseek.DeepSeekConfig;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
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
 * Example: tiny DeepSeek training on a small text file using byte-level tokens.
 *
 * <p>Architecture differences from the Llama example:
 * <ul>
 *   <li>Multi-Head Latent Attention (MLA) compresses Q through a low-rank Q bottleneck
 *       ({@code qRank}) and K/V through a KV bottleneck ({@code kvRank}), significantly
 *       reducing KV-cache memory during inference.</li>
 *   <li>All other components (RMSNorm, RoPE, SwiGLU, token embedding) are identical
 *       to the Llama architecture.</li>
 * </ul>
 *
 * Intended as a smoke test / reference, not for serious training.
 */
public final class TrainSmallDeepSeek {

    public static void main(String[] args) throws Exception {

        // ── Backend ───────────────────────────────────────────────────────────
        MetalBackend metal = new MetalBackend();
        Tensor.setBackend(metal);

        // ── Data ──────────────────────────────────────────────────────────────
        Path corpus = Path.of("sample_data/llm_training_dataset_1227_examples.txt");

        Tokenizer tok = new ByteTokenizer();
        TextDataset ds = TextDataset.fromFile(corpus, tok, 256, 123);

        // ── Model ─────────────────────────────────────────────────────────────
        int dModel = 512;
        DeepSeekConfig cfg = new DeepSeekConfig(
                tok.vocabSize(),
                256,         // maxSeqLen
                dModel,      // dModel
                4,           // nHeads
                5,           // nLayers
                1024,        // dFF
                dModel / 2,  // qRank  – Q latent dimension
                dModel / 4   // kvRank – KV latent dimension (controls KV-cache size)
        );

        DeepSeekModel model = new DeepSeekModel(cfg, 42);

        // ── Training ──────────────────────────────────────────────────────────
        Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-4);

        // ── Checkpointing ─────────────────────────────────────────────────────
        Path checkpointDir = Path.of("checkpoints");

        Trainer.StepHook checkpointHook = (step, loss, ema) -> {
            if (step > 0 && step % 500 == 0) {
                model.save(checkpointDir.resolve("small-deepseek-" + step + ".bin"));
            }
        };

        trainer.train(
                10_000_000,    // maxSteps
                2,             // batchSize
                1,             // logEvery
                0.98,          // emaBeta
                0.01,          // targetEmaLoss
                25,            // releaseEverySteps – free orphaned GPU buffers every N steps
                checkpointHook // called after each step
        );

        Path finalModelPath = checkpointDir.resolve("small-deepseek-final.bin");
        model.save(finalModelPath);

        // ── Inference ─────────────────────────────────────────────────────────
        DeepSeekModel loadedModel = new DeepSeekModel(cfg, 42);
        loadedModel.load(finalModelPath);

        String prompt = "Bob Marley was ";
        String out = TextGenerator.generate(
                loadedModel,   // model
                tok,           // tokenizer
                cfg,           // config
                prompt,        // prompt text
                200,           // maxNewTokens
                0.1,           // temperature
                20,            // topK
                1234L          // seed
        );

        System.out.println("\n=== Generated ===");
        System.out.println(out);
    }
}

