package io.github.kirstenali.deepj.chatui;

final class GeneratedText {

    private GeneratedText() {}

    static String continuation(String prompt, String generated) {
        return generated.startsWith(prompt) ? generated.substring(prompt.length()) : generated;
    }
}
