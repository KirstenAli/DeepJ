package io.github.kirstenali.deepj.models.prism;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DeepJPrismParameterCountTest {

    @Test
    void matchesAllocatedParametersForSmallModel() {
        var config = new DeepJPrismConfig(300, 8, 8, 2, 2, 16, 4, 2);
        var model = new DeepJPrismModel(config, 42L);
        long allocated = model.parameters().stream()
                .mapToLong(parameter -> parameter.value.data.length).sum();

        assertEquals(allocated, DeepJPrismParameterCount.count(config));
    }
}
