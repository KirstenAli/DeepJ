package io.github.kirstenali.deepj.models.deepseek;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DeepSeekParameterCountTest {

    @Test
    void matchesAllocatedParametersForSmallModel() {
        var config = new DeepSeekConfig(300, 8, 8, 2, 2, 16, 4, 2);
        var model = new DeepSeekModel(config, 42L);
        long allocated = model.parameters().stream()
                .mapToLong(parameter -> parameter.value.data.length).sum();

        assertEquals(allocated, DeepSeekParameterCount.count(config));
    }
}
