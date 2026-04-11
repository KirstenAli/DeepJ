package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.Trainer;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Example: tiny GPT training on a small text file using a BPE tokenizer.
 *
 * <p>Differences from the byte-level example:
 * <ul>
 *   <li>A BPE model is trained from the corpus before GPT training begins.</li>
 *   <li>The tokenizer is saved to disk so it can be reloaded alongside the model weights.</li>
 *   <li>GPT vocab size is set from the BPE vocab rather than the fixed 256-byte vocab.</li>
 * </ul>
 */
public final class TrainSmallGPTWithBPE {

    // Target BPE vocab size — must be > 257 (256 bytes + EOW) plus any special tokens.
    private static final int BPE_VOCAB_SIZE = 4_000;

    public static void main(String[] args) throws Exception {

        // ── Backend ───────────────────────────────────────────────────────────
        MetalBackend metal = new MetalBackend();
        Tensor.setBackend(metal);

        // ── Paths ─────────────────────────────────────────────────────────────
        Path corpus         = Path.of("sample_data/llm_training_dataset_1227_examples.txt");
        Path checkpointDir  = Path.of("checkpoints");
        Path tokenizerPath  = checkpointDir.resolve("small-gpt-bpe.tokenizer");

        // ── BPE tokenizer ─────────────────────────────────────────────────────
        // Train a production tokenizer that reserves <BOS>, <EOS>, and <PAD>.
        System.out.println("Training BPE tokenizer (vocab=" + BPE_VOCAB_SIZE + ") …");
        BPETokenizer tok = new BPETrainer().trainTokenizerWithDefaults(
                Files.readString(corpus), // corpus text
                BPE_VOCAB_SIZE                          // target vocab size
        );

        // Persist the tokenizer so it can be reloaded later without retraining.
        BPEModelIO.save(tokenizerPath, tok.model());
        System.out.println("Tokenizer saved → " + tokenizerPath);

        // ── Data ──────────────────────────────────────────────────────────────
        TextDataset ds = TextDataset.fromFile(
                corpus,
                tok,   // BPE tokenizer
                256,   // seqLen — context window
                123L   // seed
        );

        // ── Model ─────────────────────────────────────────────────────────────
        GPTConfig cfg = new GPTConfig(
                tok.vocabSize(), // vocab size comes from the trained BPE model
                256,             // maxSeqLen
                512,             // dModel  — embedding / hidden dimension
                4,               // nHeads  — attention heads
                5,               // nLayers — transformer blocks
                1024             // dFF     — feed-forward inner dimension
        );

        GPTModel model = new GPTModel(cfg, 42);

        // ── Training ──────────────────────────────────────────────────────────
        Trainer trainer = CausalLMTraining.trainer(model, ds, 1e-4f);

        Trainer.StepHook checkpointHook = (step, loss, ema) -> {
            if (step > 0 && step % 500 == 0) {
                model.save(checkpointDir.resolve("small-gpt-bpe-" + step + ".bin"));
            }
        };

        trainer.train(
                10_000_000,    // maxSteps
                2,             // batchSize
                1,             // logEvery
                0.98f,         // emaBeta
                0.01f,         // targetEmaLoss
                25,            // releaseEverySteps — free orphaned GPU buffers every N steps
                checkpointHook // called after each step
        );

        Path finalModelPath = checkpointDir.resolve("small-gpt-bpe-final.bin");
        model.save(finalModelPath);

        // ── Inference ─────────────────────────────────────────────────────────
        // Reload both the tokenizer and model weights from disk.
        BPETokenizer loadedTok   = new BPETokenizer(BPEModelIO.load(tokenizerPath));
        GPTModel     loadedModel = new GPTModel(cfg, 42);
        loadedModel.load(finalModelPath);

        String prompt = "<BOS> Bob Marley was ";
        String out = TextGenerator.generate(
                loadedModel,   // model
                loadedTok,     // reloaded BPE tokenizer
                cfg,           // config
                prompt,        // prompt text — leading <BOS> is encoded as a single token
                200,           // maxNewTokens
                0.1f,          // temperature
                20,            // topK
                1234L          // seed
        );

        System.out.println("\n=== Generated ===");
        System.out.println(out);
    }
}

