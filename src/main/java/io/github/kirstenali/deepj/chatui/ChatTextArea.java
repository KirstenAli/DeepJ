package io.github.kirstenali.deepj.chatui;

import javafx.application.Platform;
import javafx.scene.control.TextArea;
import javafx.scene.text.Text;

final class ChatTextArea extends TextArea {

    private static final double MAX_WIDTH = 680;
    private static final double MIN_WIDTH = 72;
    private static final double USER_PADDING = 30;
    private static final double USER_HEIGHT = 22;
    private static final double TEXT_HEIGHT = 8;

    private final boolean user;
    private boolean resizePending;

    ChatTextArea(String text, boolean user) {
        super(text);
        this.user = user;
        configure();
        requestResize();
    }

    private void configure() {
        setEditable(false);
        setWrapText(true);
        setMinHeight(USE_PREF_SIZE);
        setMaxHeight(USE_PREF_SIZE);
        setMaxWidth(MAX_WIDTH);
        getStyleClass().addAll("chat-text", user ? "user-message" : "assistant-message");
        textProperty().addListener((value, oldText, newText) -> requestResize());
        fontProperty().addListener((value, oldFont, newFont) -> requestResize());
        widthProperty().addListener((value, oldWidth, newWidth) -> requestResize());
    }

    private void requestResize() {
        if (resizePending) return;
        resizePending = true;
        Platform.runLater(this::resizeToText);
    }

    private void resizeToText() {
        resizePending = false;
        double width = preferredWidth();
        setPrefWidth(width);
        setPrefHeight(textHeight(layoutWidth(width)));
    }

    private double preferredWidth() {
        if (!user) return MAX_WIDTH;
        return clamp(unwrappedWidth() + USER_PADDING, MIN_WIDTH, MAX_WIDTH);
    }

    private double unwrappedWidth() {
        Text measure = measure();
        return measure.getLayoutBounds().getWidth();
    }

    private double layoutWidth(double preferred) {
        if (getWidth() <= 0) return preferred;
        return Math.min(preferred, getWidth());
    }

    private double textHeight(double width) {
        Text measure = measure();
        measure.setWrappingWidth(Math.max(MIN_WIDTH, width - horizontalPadding()));
        return Math.ceil(measure.getLayoutBounds().getHeight()) + verticalPadding();
    }

    private Text measure() {
        Text measure = new Text(getText().isEmpty() ? " " : getText());
        measure.setFont(getFont());
        return measure;
    }

    private double horizontalPadding() {
        return user ? USER_PADDING : 0;
    }

    private double verticalPadding() {
        return user ? USER_HEIGHT : TEXT_HEIGHT;
    }

    private static double clamp(double value, double minimum, double maximum) {
        return Math.max(minimum, Math.min(value, maximum));
    }
}
