package io.github.kirstenali.deepj.chatui;

import javafx.geometry.Pos;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;

record ChatMessageView(HBox row, Label content) {

    static ChatMessageView create(ChatMessage message) {
        return message.user() ? user(message.text()) : assistant(message.text());
    }

    private static ChatMessageView user(String text) {
        HBox row = row(Pos.CENTER_RIGHT, "user-message-row");
        Label content = content(text, "user-message");
        row.getChildren().addAll(spacer(), content);
        return new ChatMessageView(row, content);
    }

    private static ChatMessageView assistant(String text) {
        HBox row = row(Pos.TOP_LEFT, "assistant-message-row");
        Label content = content(text, "assistant-message");
        row.getChildren().addAll(avatar(), assistantBody(content), spacer());
        return new ChatMessageView(row, content);
    }

    private static HBox row(Pos alignment, String style) {
        HBox row = new HBox(14);
        row.setAlignment(alignment);
        row.getStyleClass().addAll("message-row", style);
        return row;
    }

    private static Label content(String text, String style) {
        Label content = new Label(text);
        content.setWrapText(true);
        content.setMaxWidth(680);
        content.getStyleClass().add(style);
        return content;
    }

    private static VBox assistantBody(Label content) {
        Label name = new Label("DeepJ");
        name.getStyleClass().add("assistant-name");
        VBox body = new VBox(5, name, content);
        HBox.setHgrow(body, Priority.ALWAYS);
        return body;
    }

    private static Label avatar() {
        Label avatar = new Label("D");
        avatar.getStyleClass().add("assistant-avatar");
        return avatar;
    }

    private static Region spacer() {
        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        return spacer;
    }
}
