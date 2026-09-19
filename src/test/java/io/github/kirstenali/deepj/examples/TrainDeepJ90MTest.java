package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.deepseek.DeepSeekParameterCount;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TrainDeepJ90MTest {

    @Test
    void defaultArchitectureHasExpectedParameterCount() {
        var config = TrainDeepJ90M.configuration();
        long count = DeepSeekParameterCount.count(config.modelConfig(TrainDeepJ90M.VOCAB_SIZE));

        assertEquals(90_128_896L, count);
        assertEquals(1_024, config.architecture().sequenceLength());
        assertEquals(16_384, config.tokenizer().vocabSize());
    }

    @Test
    void pretrainingCoversChinchillaTokenBudget() {
        var config = TrainDeepJ90M.configuration();
        long tokens = (long) config.training().steps() * config.training().batchSize()
                * config.architecture().sequenceLength();

        assertEquals(1_800_000_512L, tokens);
    }
}
