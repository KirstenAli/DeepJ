package io.github.kirstenali.deepj.chatui;

import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.shape.SVGPath;

record ChatMessageView(HBox row, TextArea content) {

    static ChatMessageView create(ChatMessage message) {
        return message.user() ? user(message.text()) : assistant(message.text());
    }

    private static ChatMessageView user(String text) {
        HBox row = row(Pos.CENTER_RIGHT, "user-message-row");
        TextArea content = content(text, true);
        row.getChildren().addAll(spacer(), content);
        return new ChatMessageView(row, content);
    }

    private static ChatMessageView assistant(String text) {
        HBox row = row(Pos.TOP_LEFT, "assistant-message-row");
        TextArea content = content(text, false);
        row.getChildren().addAll(avatar(), assistantBody(content), spacer());
        return new ChatMessageView(row, content);
    }

    private static HBox row(Pos alignment, String style) {
        HBox row = new HBox(14);
        row.setAlignment(alignment);
        row.getStyleClass().addAll("message-row", style);
        return row;
    }

    private static TextArea content(String text, boolean user) {
        return new ChatTextArea(text, user);
    }

    private static VBox assistantBody(TextArea content) {
        Label name = new Label("DeepJ");
        name.getStyleClass().add("assistant-name");
        VBox body = new VBox(5, name, content, copyButton(content));
        HBox.setHgrow(body, Priority.ALWAYS);
        return body;
    }

    private static Button copyButton(TextArea content) {
        Button button = new Button();
        button.setAccessibleText("Copy response");
        button.setGraphic(copyIcon());
        button.setTooltip(new Tooltip("Copy response"));
        button.setOnAction(event -> ChatClipboard.copy(content.getText()));
        button.getStyleClass().add("copy-message-button");
        return button;
    }

    private static SVGPath copyIcon() {
        SVGPath icon = new SVGPath();
        icon.setContent("M6 5h8v10H6z M3 12V2h8");
        icon.getStyleClass().add("button-icon");
        return icon;
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
