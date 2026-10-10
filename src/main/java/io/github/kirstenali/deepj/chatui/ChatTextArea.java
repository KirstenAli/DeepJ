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
    private static final double USER_HEIGHT = 22;
    private static final double ASSISTANT_HEIGHT = 0;
    private static final double WRAP_GUTTER = 14;
    private static final double HEIGHT_SAFETY = 2;

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
        setPrefHeight(textHeight(layoutWidth(width)));
        disableInternalScroll();
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

    private double textHeight(double width) {
        Text measure = measure();
        measure.setWrappingWidth(Math.max(MIN_WIDTH, width - horizontalPadding() - WRAP_GUTTER));
        return Math.ceil(measure.getLayoutBounds().getHeight()) + verticalPadding() + HEIGHT_SAFETY;
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

    private double horizontalPadding() {
        return user ? USER_PADDING : 0;
    }

    private double verticalPadding() {
        return user ? USER_HEIGHT : ASSISTANT_HEIGHT;
    }

    private void disableInternalScroll() {
        if (!(lookup(".scroll-pane") instanceof ScrollPane scroll)) return;
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        scroll.setVbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        scroll.setPannable(false);
    }

    private void resetScroll() {
        setScrollLeft(0);
        setScrollTop(0);
    }

    private void scrollConversation(ScrollEvent event) {
        ScrollPane scroll = conversationScroll();
        if (scroll != null && scrollOverflow(scroll) > 0) scrollOuter(scroll, event);
        event.consume();
    }

    private void scrollOuter(ScrollPane scroll, ScrollEvent event) {
        double value = scroll.getVvalue() - event.getDeltaY() / scrollOverflow(scroll);
        scroll.setVvalue(clamp(value, 0, 1));
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
