package io.github.kirstenali.deepj.chatui;

import java.util.Objects;

import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;

record ChatGenerationSettings(int maxTokens, double temperature, int topK, long seed,
                              double repetitionPenalty, ContextWindowMode contextWindow) {

    ChatGenerationSettings(int maxTokens, double temperature, int topK, long seed, double repetitionPenalty) {
        this(maxTokens, temperature, topK, seed, repetitionPenalty, ContextWindowMode.REBUILD);
    }

    ChatGenerationSettings {
        Objects.requireNonNull(contextWindow, "contextWindow");
        if (maxTokens < 1) throw new IllegalArgumentException("Maximum tokens must be positive");
        if (!Double.isFinite(temperature) || temperature <= 0) {
            throw new IllegalArgumentException("Temperature must be positive");
        }
        if (topK < 0) throw new IllegalArgumentException("Top-k cannot be negative");
        if (!Double.isFinite(repetitionPenalty) || repetitionPenalty < 1.0) {
            throw new IllegalArgumentException("Repetition penalty must be at least 1");
        }
    }

    ChatGenerationSettings withPenalty() {
        double penalty = repetitionPenalty == 2.0 ? 1.1 : Math.min(2.0, repetitionPenalty + 0.1);
        return new ChatGenerationSettings(maxTokens, temperature, topK, seed, penalty, contextWindow);
    }

    ChatGenerationSettings withNewSeed() {
        return new ChatGenerationSettings(maxTokens, temperature, topK, seed + 1, repetitionPenalty, contextWindow);
    }
}
