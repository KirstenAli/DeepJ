package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TrainDeepSeekKnowledgeTest {

    @Test
    void defaultArchitectureHasExpectedParameterCount() {
        var config = TrainDeepSeekKnowledge.configuration().modelConfig(
                TrainDeepSeekKnowledge.VOCAB_SIZE);
        var model = new DeepSeekModel(config, 42L);

        long parameters = model.parameters().stream()
                .mapToLong(parameter -> parameter.value.data.length)
                .sum();

        assertEquals(19_006_848L, parameters);
    }

    @Test
    void releasesMetalMemoryAfterEveryStepByDefault() {
        var training = TrainDeepSeekKnowledge.configuration().training();
        assertEquals(TrainDeepSeekKnowledge.RELEASE_EVERY, training.releaseEvery());
        assertEquals(1, training.releaseEvery());
    }
}
