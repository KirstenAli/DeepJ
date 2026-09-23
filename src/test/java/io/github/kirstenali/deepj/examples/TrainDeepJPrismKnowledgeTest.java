package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.prism.DeepJPrismModel;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TrainDeepJPrismKnowledgeTest {

    @Test
    void defaultArchitectureHasExpectedParameterCount() {
        var config = TrainDeepJPrismKnowledge.configuration().modelConfig(
                TrainDeepJPrismKnowledge.VOCAB_SIZE);
        var model = new DeepJPrismModel(config, 42L);

        long parameters = model.parameters().stream()
                .mapToLong(parameter -> parameter.value.data.length)
                .sum();

        assertEquals(19_006_848L, parameters);
    }

    @Test
    void usesStandardResourceReleaseInterval() {
        var training = TrainDeepJPrismKnowledge.configuration().training();
        assertEquals(TrainDeepJPrismKnowledge.RELEASE_EVERY, training.releaseEvery());
        assertEquals(25, training.releaseEvery());
    }
}
