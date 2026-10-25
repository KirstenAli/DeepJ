package io.github.kirstenali.deepj.cli;

import java.util.Locale;

final class ChatCommands {
    static final String[] NAMES = {"/help", "/settings", "/set", "/alternatives", "/theme", "/clear", "/exit"};

    private ChatCommands() {}

    static boolean run(ChatSession session, String input) {
        String[] words = input.strip().split("\\s+");
        switch (words[0]) {
            case "/help" -> session.help();
            case "/settings" -> session.settings();
            case "/clear" -> session.clear();
            case "/alternatives" -> session.alternatives();
            case "/theme" -> theme(session, words);
            case "/set" -> edit(session, words);
            case "/exit" -> { return false; }
            default -> throw new IllegalArgumentException("Unknown command. Use /help, or // for a literal slash.");
        }
        return true;
    }

    private static void theme(ChatSession session, String[] words) {
        require(words.length == 2, "Use /theme aurora, /theme ember or /theme mono.");
        try {
            session.theme(CliTheme.valueOf(words[1].toUpperCase(Locale.ROOT)));
        } catch (IllegalArgumentException invalid) {
            throw new IllegalArgumentException("Choose aurora, ember or mono.", invalid);
        }
    }

    private static void edit(ChatSession session, String[] words) {
        require(words.length == 3, "Use /set <setting> <value>. See /settings.");
        session.edit(words[1], words[2]);
        session.settings();
    }

    private static void require(boolean valid, String message) {
        if (!valid) throw new IllegalArgumentException(message);
    }
}
