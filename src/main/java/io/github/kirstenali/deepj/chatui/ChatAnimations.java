package io.github.kirstenali.deepj.chatui;

import javafx.animation.Animation;
import javafx.animation.FadeTransition;
import javafx.animation.Interpolator;
import javafx.animation.KeyFrame;
import javafx.animation.KeyValue;
import javafx.animation.ParallelTransition;
import javafx.animation.Timeline;
import javafx.animation.TranslateTransition;
import javafx.scene.Node;
import javafx.scene.layout.Region;
import javafx.util.Duration;

final class ChatAnimations {

    private static final Duration SPEED = Duration.millis(180);
    private static final double DRAWER_OFFSET = 24;
    private static final double SIDEBAR_WIDTH = 270;

    private ChatAnimations() {
    }

    static Animation drawer(Node panel, Node backdrop, boolean opening) {
        prepareDrawer(panel, backdrop, opening);
        ParallelTransition motion = new ParallelTransition(
                fade(panel, opening), fade(backdrop, opening), slide(panel, opening));
        if (!opening) motion.setOnFinished(event -> hide(panel, backdrop));
        return motion;
    }

    static Animation sidebar(Region sidebar, boolean opening) {
        prepareSidebar(sidebar, opening);
        Timeline motion = new Timeline(frame(sidebar, opening));
        if (!opening) motion.setOnFinished(event -> hide(sidebar));
        return motion;
    }

    private static void prepareDrawer(Node panel, Node backdrop, boolean opening) {
        if (!opening) return;
        show(panel, backdrop);
        panel.setOpacity(0);
        panel.setTranslateX(DRAWER_OFFSET);
        backdrop.setOpacity(0);
    }

    private static void prepareSidebar(Region sidebar, boolean opening) {
        if (opening) show(sidebar);
        resize(sidebar, opening ? 0 : sidebar.getWidth());
        sidebar.setOpacity(opening ? 0 : 1);
    }

    private static FadeTransition fade(Node node, boolean opening) {
        FadeTransition fade = new FadeTransition(SPEED, node);
        fade.setToValue(opening ? 1 : 0);
        fade.setInterpolator(Interpolator.EASE_BOTH);
        return fade;
    }

    private static TranslateTransition slide(Node node, boolean opening) {
        TranslateTransition slide = new TranslateTransition(SPEED, node);
        slide.setToX(opening ? 0 : DRAWER_OFFSET);
        slide.setInterpolator(Interpolator.EASE_BOTH);
        return slide;
    }

    private static KeyFrame frame(Region sidebar, boolean opening) {
        double width = opening ? SIDEBAR_WIDTH : 0;
        return new KeyFrame(SPEED, width(sidebar.minWidthProperty(), width),
                width(sidebar.prefWidthProperty(), width), width(sidebar.maxWidthProperty(), width),
                new KeyValue(sidebar.opacityProperty(), opening ? 1 : 0, Interpolator.EASE_BOTH));
    }

    private static KeyValue width(javafx.beans.property.DoubleProperty property, double value) {
        return new KeyValue(property, value, Interpolator.EASE_BOTH);
    }

    private static void resize(Region region, double width) {
        region.setMinWidth(width);
        region.setPrefWidth(width);
        region.setMaxWidth(width);
    }

    private static void show(Node... nodes) {
        for (Node node : nodes) {
            setShowing(node, true);
        }
    }

    private static void hide(Node... nodes) {
        for (Node node : nodes) {
            setShowing(node, false);
        }
    }

    private static void setShowing(Node node, boolean showing) {
        node.setManaged(showing);
        node.setVisible(showing);
    }
}
