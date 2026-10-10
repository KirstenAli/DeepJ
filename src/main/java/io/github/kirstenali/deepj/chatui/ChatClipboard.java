package io.github.kirstenali.deepj.chatui;

import javafx.scene.input.Clipboard;
import javafx.scene.input.ClipboardContent;

final class ChatClipboard {

    private ChatClipboard() {
    }

    static void copy(String text) {
        ClipboardContent content = new ClipboardContent();
        content.putString(text);
        Clipboard.getSystemClipboard().setContent(content);
    }
}
