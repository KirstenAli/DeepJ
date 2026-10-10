package io.github.kirstenali.deepj.chatui;

record GenerationSettings(int maxTokens, float temperature, int topK, long seed) {

    GenerationSettings {
        if (maxTokens <= 0) throw new IllegalArgumentException("Max tokens must be greater than zero");
        if (!Float.isFinite(temperature) || temperature <= 0.0f) {
            throw new IllegalArgumentException("Temperature must be greater than zero");
        }
        if (topK <= 0) throw new IllegalArgumentException("Top-k must be greater than zero");
    }

    static GenerationSettings parse(String maxTokens, String temperature,
                                    String topK, String seed) {
        return new GenerationSettings(integer(maxTokens, "Max tokens"),
                decimal(temperature, "Temperature"), integer(topK, "Top-k"),
                wholeNumber(seed, "Seed"));
    }

    private static int integer(String value, String name) {
        try {
            return Integer.parseInt(value.trim());
        } catch (RuntimeException error) {
            throw new IllegalArgumentException(name + " must be an integer");
        }
    }

    private static long wholeNumber(String value, String name) {
        try {
            return Long.parseLong(value.trim());
        } catch (RuntimeException error) {
            throw new IllegalArgumentException(name + " must be an integer");
        }
    }

    private static float decimal(String value, String name) {
        try {
            return Float.parseFloat(value.trim());
        } catch (RuntimeException error) {
            throw new IllegalArgumentException(name + " must be a number");
        }
    }
}
