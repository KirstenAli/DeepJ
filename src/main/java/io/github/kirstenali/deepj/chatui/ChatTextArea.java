package io.github.kirstenali.deepj.chatui;

import javafx.application.Platform;
import javafx.scene.Parent;
import javafx.scene.control.TextArea;
import javafx.scene.control.ScrollPane;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.Region;
import javafx.scene.text.Text;

final class ChatTextArea extends TextArea {

    private static final double MAX_WIDTH = 680;
    private static final double MIN_WIDTH = 72;
    private static final double USER_PADDING = 30;
    private static final double MIN_TEXT_WIDTH = 24;
    private static final double WIDTH_GUTTER = 14;
    private static final double HEIGHT_GUTTER = 2;

    private final boolean user;
    private Text observedText;
    private boolean resizePending;
    private boolean settlePending;

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

    @Override
    protected void layoutChildren() {
        super.layoutChildren();
        fitRenderedText();
        resetScroll();
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
        setPrefRowCount(1);
        setPrefHeight(USE_COMPUTED_SIZE);
        applyCss();
        layout();
        observeRenderedText();
        setPrefHeight(renderedHeight());
        fitRenderedText();
        disableInternalScroll();
        resetScroll();
    }

    private void fitRenderedText() {
        if (!(lookup(".text") instanceof Text text)) return;
        if (!(lookup(".viewport") instanceof Region viewport)) return;
        if (getHeight() <= 0 || viewport.getHeight() <= 0) return;
        double required = requiredHeight(text, viewport);
        if (required > getPrefHeight() + 1) setPrefHeight(required);
        else if (getPrefHeight() - required > 3) requestSettle();
    }

    private double requiredHeight(Text text, Region viewport) {
        double chrome = getHeight() - viewport.getHeight();
        return Math.ceil(text.getLayoutBounds().getHeight() + contentPadding() + chrome + 2);
    }

    private void requestSettle() {
        if (settlePending) return;
        settlePending = true;
        Platform.runLater(this::settleHeight);
    }

    private void observeRenderedText() {
        if (!(lookup(".text") instanceof Text text) || text == observedText) return;
        observedText = text;
        text.layoutBoundsProperty().addListener((value, oldBounds, newBounds) -> requestSettle());
    }

    private void settleHeight() {
        settlePending = false;
        if (!(lookup(".text") instanceof Text text)) return;
        if (!(lookup(".viewport") instanceof Region viewport)) return;
        if (getHeight() <= 0 || viewport.getHeight() <= 0) return;
        double required = requiredHeight(text, viewport);
        if (Math.abs(getPrefHeight() - required) > 1) setPrefHeight(required);
    }

    private double preferredWidth() {
        if (!user) return MAX_WIDTH;
        return clamp(unwrappedWidth() + USER_PADDING + WIDTH_GUTTER, MIN_WIDTH, MAX_WIDTH);
    }

    private double unwrappedWidth() {
        Text measure = measure();
        return measure.getLayoutBounds().getWidth();
    }

    private double renderedHeight() {
        Text rendered = measure();
        rendered.setWrappingWidth(contentWidth());
        double height = Math.max(lineHeight(), rendered.getLayoutBounds().getHeight());
        return height + contentPadding() + HEIGHT_GUTTER;
    }

    private double contentWidth() {
        if (!(lookup(".viewport") instanceof Region viewport)) return MIN_TEXT_WIDTH;
        return Math.max(MIN_TEXT_WIDTH, viewport.getWidth() - horizontalContentPadding());
    }

    private double contentPadding() {
        if (!(lookup(".content") instanceof Region content)) return 0;
        return content.getInsets().getTop() + content.getInsets().getBottom();
    }

    private double horizontalContentPadding() {
        if (!(lookup(".content") instanceof Region content)) return 0;
        return content.getInsets().getLeft() + content.getInsets().getRight();
    }

    private double lineHeight() {
        Text line = new Text("Ay");
        line.setFont(getFont());
        return line.getLayoutBounds().getHeight();
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
