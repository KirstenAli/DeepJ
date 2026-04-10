package io.github.kirstenali.deepj.tokenizers.bpe;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class BPEModelIOTest {

    @Test
    void saveLoad_roundTripsModelAndTokenizerBehavior() throws IOException {
        BPETrainer trainer = new BPETrainer();
        BPEModel model = trainer.train("hello hello world", 280, List.of("<BOS>", "<EOS>", "<PAD>"));
        BPETokenizer original = new BPETokenizer(model);

        Path tmp = Files.createTempFile("deepj-bpe", ".bin");
        try {
            BPEModelIO.save(tmp, model);
            BPEModel loaded = BPEModelIO.load(tmp);
            BPETokenizer reloaded = new BPETokenizer(loaded);

            String input = "<BOS> hello world <EOS>";
            assertArrayEquals(original.encode(input), reloaded.encode(input));
            assertEquals(original.decode(original.encode(input)), reloaded.decode(reloaded.encode(input)));
            assertEquals(model.specialTokenToId(), loaded.specialTokenToId());
            assertEquals(model.merges(), loaded.merges());
            assertEquals(model.mergeToNewId(), loaded.mergeToNewId());
        } finally {
            Files.deleteIfExists(tmp);
        }
    }
}

