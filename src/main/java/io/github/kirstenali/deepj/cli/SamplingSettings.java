package io.github.kirstenali.deepj.cli;

record SamplingSettings(int maxTokens, float temperature, int topK, long seed, float repetitionPenalty) {
    SamplingSettings {
        require(maxTokens > 0, "max-tokens", "must be positive");
        require(Float.isFinite(temperature) && temperature > 0, "temperature", "must be finite and positive");
        require(topK >= 0, "top-k", "cannot be negative; use 0 for the full vocabulary");
        require(Float.isFinite(repetitionPenalty) && repetitionPenalty >= 1,
                "repetition-penalty", "must be finite and at least 1");
    }

    SamplingSettings edit(String name, String value) {
        return switch (name) {
            case "max-tokens" -> new SamplingSettings(Integer.parseInt(value), temperature, topK, seed,
                    repetitionPenalty);
            case "temperature" -> new SamplingSettings(maxTokens, Float.parseFloat(value), topK, seed,
                    repetitionPenalty);
            case "top-k" -> new SamplingSettings(maxTokens, temperature, Integer.parseInt(value), seed,
                    repetitionPenalty);
            case "seed" -> new SamplingSettings(maxTokens, temperature, topK, Long.parseLong(value), repetitionPenalty);
            case "repetition-penalty" -> new SamplingSettings(maxTokens, temperature, topK, seed,
                    Float.parseFloat(value));
            default -> throw new IllegalArgumentException("Unknown setting: " + name + ". Use /settings.");
        };
    }

    SamplingSettings alternativePenalty() {
        float penalty = repetitionPenalty == 2.0f ? 1.1f : Math.min(2.0f, repetitionPenalty + 0.1f);
        return new SamplingSettings(maxTokens, temperature, topK, seed, penalty);
    }

    SamplingSettings alternativeSeed() {
        return new SamplingSettings(maxTokens, temperature, topK, seed + 1, repetitionPenalty);
    }

    private static void require(boolean valid, String option, String condition) {
        if (!valid) throw new IllegalArgumentException(option + " " + condition);
    }
}
