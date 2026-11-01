package io.github.kirstenali.deepj.nn.transformer.prism;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DeepJPrismParameterCountTest {

    @Test
    void matchesAllocatedParametersForSmallModel() {
        var config = new DeepJPrismConfig(300, 8, 8, 2, 2, 16, 4, 2);
        var model = new DeepJPrism(config, 42L);
        long allocated = model.parameters().stream()
                .mapToLong(parameter -> parameter.value.rawData().length).sum();

        assertEquals(allocated, DeepJPrismParameterCount.count(config));
    }
}
