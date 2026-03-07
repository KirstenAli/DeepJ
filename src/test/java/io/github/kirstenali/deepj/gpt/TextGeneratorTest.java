
package io.github.kirstenali.deepj.gpt;

import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.models.gpt.TextGenerator;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TextGeneratorTest {

    @Test
    void generate_runsAndReturnsNonEmptyString() {
        Tokenizer tok = new ByteTokenizer();

        GPTConfig cfg = new GPTConfig(
                ByteTokenizer.VOCAB_SIZE,
                16,   // maxSeqLen
                32,   // dModel
                4,    // nHeads
                2,    // nLayers
                64    // dFF
        );

        GPTModel model = new GPTModel(cfg, 1L);
        String out = TextGenerator.generate(model, tok, cfg, "hi", 8, 1.0, 0, 2L);

        Assertions.assertNotNull(out);
        Assertions.assertTrue(out.length() >= 2);
    }
}
