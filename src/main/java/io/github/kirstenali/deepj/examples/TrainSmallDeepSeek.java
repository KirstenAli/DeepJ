package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.deepseek.DeepSeekConfig;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.nio.file.Path;

/**
 * Example: tiny DeepSeek training on a small text file using byte-level tokens.
 *
 * <p>Architecture differences from the Llama example:
 * <ul>
 *   <li>Multi-Head Latent Attention (MLA) compresses Q through a low-rank Q bottleneck
 *       ({@code qRank}) and K/V through a KV bottleneck ({@code kvRank}).</li>
 *   <li>All other components (RMSNorm, RoPE, SwiGLU, token embedding) are identical
 *       to the Llama architecture.</li>
 * </ul>
 *
 * Intended as a smoke test / reference, not for serious training.
 */
public final class TrainSmallDeepSeek {

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        Tokenizer tok = new ByteTokenizer();
        TextDataset ds = TrainingExampleSupport.dataset(tok);
        int dModel = 512;
        DeepSeekConfig cfg = new DeepSeekConfig(tok.vocabSize(), 256, dModel, 4, 5,
                1024, dModel / 2, dModel / 4);
        DeepSeekModel model = new DeepSeekModel(cfg, 42);
        Path finalModelPath = TrainingExampleSupport.trainAndSave(model, model, ds, "small-deepseek");
        DeepSeekModel loadedModel = new DeepSeekModel(cfg, 42);
        loadedModel.load(finalModelPath);
        TrainingExampleSupport.generate(loadedModel, tok, cfg, "Bob Marley was ");
    }
}
