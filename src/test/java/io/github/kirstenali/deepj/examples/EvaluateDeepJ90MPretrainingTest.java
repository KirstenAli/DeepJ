package io.github.kirstenali.deepj.examples;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

class EvaluateDeepJ90MPretrainingTest {

    @Test
    void defaultsUseFineWebPretrainingCorpus() {
        var settings = EvaluateDeepJ90MPretraining.Settings.fromSystemProperties();
        assertEquals(Path.of("checkpoints/deepj-90m/pretrain"), settings.base());
        assertEquals(Path.of("sample_data/deepj-90m/fineweb-edu.txt"), settings.corpus());
        assertEquals(10, settings.batches());
        assertEquals(1_000_090L, settings.seed());
    }
}
