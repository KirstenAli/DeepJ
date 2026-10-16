package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.models.PromptFormat;
import io.github.kirstenali.deepj.models.TransformerConfig;

import java.util.Objects;

public record ModelSettings(TransformerConfig config, PromptFormat promptFormat) {

    public ModelSettings {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(promptFormat, "promptFormat");
    }
}
