package io.github.kirstenali.deepj.chatui;

import javafx.concurrent.Worker;
import javafx.scene.input.Clipboard;
import javafx.scene.input.ClipboardContent;
import javafx.scene.layout.StackPane;
import javafx.scene.web.WebView;
import netscape.javascript.JSObject;

import java.util.ArrayList;
import java.util.List;

public final class ChatTranscript extends StackPane {

    private final WebView view = new WebView();
    private final List<ChatMessage> messages = new ArrayList<>();
    private final ClipboardBridge clipboardBridge = new ClipboardBridge();
    private boolean ready;
    private String theme = "light";

    public ChatTranscript() {
        getChildren().add(view);
        view.setContextMenuEnabled(true);
        view.getEngine().getLoadWorker().stateProperty().addListener((ob, old, state) -> {
            if (state == Worker.State.SUCCEEDED) showLoadedPage();
        });
        view.getEngine().load(getClass().getResource("/chat-transcript.html").toExternalForm());
    }

    public int add(boolean user, String text) {
        int index = messages.size();
        messages.add(new ChatMessage(text, user));
        if (ready) render(index);
        return index;
    }

    public void update(int index, String text) {
        ChatMessage old = messages.get(index);
        messages.set(index, new ChatMessage(text, old.user()));
        if (ready) render(index);
    }

    public void clear() {
        messages.clear();
        if (ready) view.getEngine().executeScript("clearMessages()");
    }

    public void setTheme(String name) {
        theme = name;
        if (ready) window().call("setTheme", name);
    }

    private void showLoadedPage() {
        ready = true;
        window().setMember("chatBridge", clipboardBridge);
        setTheme(theme);
        for (int index = 0; index < messages.size(); index++) {
            render(index);
        }
    }

    private void render(int index) {
        ChatMessage message = messages.get(index);
        window().call("showMessage", index, message.user() ? "user" : "assistant", message.text());
    }

    private JSObject window() {
        return (JSObject) view.getEngine().executeScript("window");
    }

    public static final class ClipboardBridge {

        public void copy(String text) {
            ClipboardContent content = new ClipboardContent();
            content.putString(text);
            Clipboard.getSystemClipboard().setContent(content);
        }
    }
}
