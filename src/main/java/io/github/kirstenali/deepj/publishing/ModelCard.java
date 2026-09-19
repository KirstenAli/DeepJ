package io.github.kirstenali.deepj.publishing;

public record ModelCard(
        String name,
        String summary,
        String license,
        String language,
        String trainingData,
        String limitations
) {
    public ModelCard {
        name = requireText(name, "name");
        summary = requireText(summary, "summary");
        license = requireSlug(license, "license");
        language = requireSlug(language, "language");
        trainingData = requireText(trainingData, "trainingData");
        limitations = requireText(limitations, "limitations");
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value.strip();
    }

    private static String requireSlug(String value, String field) {
        String slug = requireText(value, field);
        if (!slug.matches("[A-Za-z0-9._-]+")) {
            throw new IllegalArgumentException(field + " must be a metadata slug");
        }
        return slug;
    }
}
