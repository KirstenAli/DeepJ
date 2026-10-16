package io.github.kirstenali.deepj.models;

public enum PromptFormat {
    PLAIN,
    INSTRUCTION_RESPONSE;

    public String apply(String prompt) {
        if (this == PLAIN || alreadyFormatted(prompt)) return prompt;
        return "Instruction:\n" + prompt.strip() + "\nResponse:\n";
    }

    private static boolean alreadyFormatted(String prompt) {
        String text = prompt.strip();
        return text.startsWith("Instruction:") && text.endsWith("Response:");
    }
}
