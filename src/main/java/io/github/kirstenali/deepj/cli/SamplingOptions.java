package io.github.kirstenali.deepj.cli;

import picocli.CommandLine.Option;
import io.github.kirstenali.deepj.models.ContextWindowMode;

final class SamplingOptions {
    @Option(names = "--max-tokens", description = "Maximum generated tokens (default: ${DEFAULT-VALUE}).")
    int maxTokens = 80;

    @Option(names = "--temperature", description = "Lower values favor likely next tokens (default: ${DEFAULT-VALUE}).")
    float temperature = 0.8f;

    @Option(names = "--top-k",
            description = "Sample from the K likeliest tokens; 0 uses all (default: ${DEFAULT-VALUE}).")
    int topK = 40;

    @Option(names = "--seed",
            description = "Sampling seed; changing it can change the response (default: ${DEFAULT-VALUE}).")
    long seed = 1234;

    @Option(names = "--repetition-penalty",
            description = "Discourage token repeats; 1 is off (default: ${DEFAULT-VALUE}).")
    float penalty = 1;

    @Option(names = "--rolling-cache", description = "Reuse a rolling KV cache beyond the window (RoPE models only).")
    boolean rollingCache;

    SamplingSettings settings() {
        return new SamplingSettings(maxTokens, temperature, topK, seed, penalty,
                rollingCache ? ContextWindowMode.ROLLING : ContextWindowMode.REBUILD);
    }
}
