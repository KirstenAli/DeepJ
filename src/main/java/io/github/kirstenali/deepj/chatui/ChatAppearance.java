package io.github.kirstenali.deepj.chatui;

import javafx.animation.Animation;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.control.Button;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;

import java.util.Map;

final class ChatAppearance {

    private final Parent root;
    private final VBox panel;
    private final Region backdrop;
    private final Map<ChatTheme, Button> buttons;
    private Animation drawerAnimation;
    private Animation sidebarAnimation;
    private boolean settingsOpen;
    private boolean sidebarOpen = true;

    ChatAppearance(Parent root, VBox panel, Region backdrop,
                   Button light, Button dark, Button midnight) {
        this.root = root;
        this.panel = panel;
        this.backdrop = backdrop;
        buttons = Map.of(ChatTheme.LIGHT, light, ChatTheme.DARK, dark,
                ChatTheme.MIDNIGHT, midnight);
    }

    void apply(ChatTheme theme) {
        theme.apply(root);
        buttons.forEach((value, button) -> select(button, value == theme));
    }

    void toggle() {
        setVisible(!settingsOpen);
    }

    void hide() {
        setVisible(false);
    }

    void toggleSidebar() {
        sidebarOpen = !sidebarOpen;
        Region sidebar = (Region) root.lookup("#sidebarPane");
        Node openButton = root.lookup("#sidebarOpenButton");
        show(openButton, !sidebarOpen);
        sidebarAnimation = play(sidebarAnimation, ChatAnimations.sidebar(sidebar, sidebarOpen));
    }

    private void setVisible(boolean visible) {
        if (settingsOpen == visible) return;
        settingsOpen = visible;
        drawerAnimation = play(drawerAnimation, ChatAnimations.drawer(panel, backdrop, visible));
    }

    private static Animation play(Animation current, Animation next) {
        if (current != null) current.stop();
        next.play();
        return next;
    }

    private static void show(Node node, boolean visible) {
        node.setManaged(visible);
        node.setVisible(visible);
    }

    private static void select(Button button, boolean selected) {
        button.getStyleClass().remove("selected");
        if (selected) button.getStyleClass().add("selected");
    }
}
