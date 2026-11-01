package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.nn.transformer.TransformerConfig;

import java.util.Objects;
import java.util.Map;

public record ModelSettings(TransformerConfig config, PromptFormat promptFormat,
                            Map<String, String> metadata) {

    public ModelSettings {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(promptFormat, "promptFormat");
        metadata = Map.copyOf(metadata);
    }

    public ModelSettings(TransformerConfig config, PromptFormat promptFormat) {
        this(config, promptFormat, Map.of());
    }
}
