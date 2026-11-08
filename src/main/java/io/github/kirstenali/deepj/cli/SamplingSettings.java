package io.github.kirstenali.deepj.cli;

import java.util.Locale;
import java.util.Objects;

import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;

record SamplingSettings(int maxTokens, float temperature, int topK, long seed,
                         float repetitionPenalty, ContextWindowMode contextWindow, boolean thinking) {
    SamplingSettings(int maxTokens, float temperature, int topK, long seed, float repetitionPenalty) {
        this(maxTokens, temperature, topK, seed, repetitionPenalty, ContextWindowMode.REBUILD);
    }

    SamplingSettings(int maxTokens, float temperature, int topK, long seed,
                     float repetitionPenalty, ContextWindowMode contextWindow) {
        this(maxTokens, temperature, topK, seed, repetitionPenalty, contextWindow, false);
    }

    SamplingSettings {
        Objects.requireNonNull(contextWindow, "contextWindow");
        require(maxTokens > 0, "max-tokens", "must be positive");
        require(Float.isFinite(temperature) && temperature > 0, "temperature", "must be finite and positive");
        require(topK >= 0, "top-k", "cannot be negative; use 0 for the full vocabulary");
        require(Float.isFinite(repetitionPenalty) && repetitionPenalty >= 1,
                "repetition-penalty", "must be finite and at least 1");
    }

    SamplingSettings edit(String name, String value) {
        return name.equals("thinking") ? withThinking(ChatCommands.toggleEnabled(name, value))
                : editSampling(name, value);
    }

    private SamplingSettings editSampling(String name, String value) {
        return switch (name) {
            case "max-tokens" -> withSampling(Integer.parseInt(value), temperature, topK, seed,
                    repetitionPenalty, contextWindow);
            case "temperature" -> withSampling(maxTokens, Float.parseFloat(value), topK,
                    seed, repetitionPenalty, contextWindow);
            case "top-k" -> withSampling(maxTokens, temperature, Integer.parseInt(value), seed,
                    repetitionPenalty, contextWindow);
            case "seed" -> withSampling(maxTokens, temperature, topK, Long.parseLong(value),
                    repetitionPenalty, contextWindow);
            case "repetition-penalty" -> withSampling(maxTokens, temperature, topK, seed,
                    Float.parseFloat(value), contextWindow);
            case "context-window" -> withSampling(maxTokens, temperature, topK, seed,
                    repetitionPenalty, ContextWindowMode.valueOf(value.toUpperCase(Locale.ROOT)));
            default -> throw new IllegalArgumentException("Unknown setting: " + name + ". Use /settings.");
        };
    }

    SamplingSettings alternativePenalty() {
        float penalty = repetitionPenalty == 2.0f ? 1.1f : Math.min(2.0f, repetitionPenalty + 0.1f);
        return withSampling(maxTokens, temperature, topK, seed, penalty, contextWindow);
    }

    SamplingSettings alternativeSeed() {
        return withSampling(maxTokens, temperature, topK, seed + 1, repetitionPenalty, contextWindow);
    }

    SamplingSettings withThinking(boolean enabled) {
        return new SamplingSettings(maxTokens, temperature, topK, seed, repetitionPenalty, contextWindow, enabled);
    }

    void checkModel(CliModel model) {
        require(!thinking || model.supportsThinking(), "thinking", "is available for Qwen3 only");
    }

    private SamplingSettings withSampling(int maxTokens, float temperature, int topK, long seed,
                                          float repetitionPenalty, ContextWindowMode contextWindow) {
        return new SamplingSettings(maxTokens, temperature, topK, seed, repetitionPenalty, contextWindow, thinking);
    }

    private static void require(boolean valid, String option, String condition) {
        if (!valid) throw new IllegalArgumentException(option + " " + condition);
    }
}
