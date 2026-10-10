package io.github.kirstenali.deepj.chatui;

import javafx.application.Platform;
import javafx.scene.Parent;
import javafx.scene.control.TextArea;
import javafx.scene.control.ScrollPane;
import javafx.scene.input.ScrollEvent;
import javafx.scene.text.Text;

final class ChatTextArea extends TextArea {

    private static final double MAX_WIDTH = 680;
    private static final double MIN_WIDTH = 72;
    private static final double USER_PADDING = 30;
    private static final double WRAP_GUTTER = 14;

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
        addEventFilter(ScrollEvent.SCROLL, this::scrollConversation);
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
        setPrefRowCount(rowCount(layoutWidth(width)));
        setPrefHeight(USE_COMPUTED_SIZE);
        resetScroll();
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

    private int rowCount(double width) {
        Text measure = measure();
        measure.setWrappingWidth(Math.max(MIN_WIDTH, width - horizontalPadding() - WRAP_GUTTER));
        return Math.max(1, (int) Math.ceil(measure.getLayoutBounds().getHeight() / lineHeight()));
    }

    private Text measure() {
        Text measure = new Text(measuredText());
        measure.setFont(getFont());
        return measure;
    }

    private String measuredText() {
        if (getText().isEmpty()) return " ";
        return getText().endsWith("\n") ? getText() + " " : getText();
    }

    private double lineHeight() {
        Text line = new Text("Ay");
        line.setFont(getFont());
        return line.getLayoutBounds().getHeight();
    }

    private double horizontalPadding() {
        return user ? USER_PADDING : 0;
    }

    private void resetScroll() {
        setScrollLeft(0);
        setScrollTop(0);
    }

    private void scrollConversation(ScrollEvent event) {
        ScrollPane scroll = conversationScroll();
        if (scroll == null || scrollOverflow(scroll) <= 0) return;
        scroll.setVvalue(clamp(scroll.getVvalue() - event.getDeltaY() / scrollOverflow(scroll), 0, 1));
        event.consume();
    }

    private ScrollPane conversationScroll() {
        Parent parent = getParent();
        while (parent != null && !(parent instanceof ScrollPane)) {
            parent = parent.getParent();
        }
        return parent instanceof ScrollPane scroll ? scroll : null;
    }

    private double scrollOverflow(ScrollPane scroll) {
        double content = scroll.getContent().getBoundsInLocal().getHeight();
        return content - scroll.getViewportBounds().getHeight();
    }

    private static double clamp(double value, double minimum, double maximum) {
        return Math.max(minimum, Math.min(value, maximum));
    }
}
