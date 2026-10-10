package io.github.kirstenali.deepj.chatui;

import javafx.scene.Parent;

public enum ChatTheme {
    LIGHT("Light", "theme-light"),
    DARK("Dark", "theme-dark"),
    MIDNIGHT("Midnight", "theme-midnight");

    private final String label;
    private final String styleClass;

    ChatTheme(String label, String styleClass) {
        this.label = label;
        this.styleClass = styleClass;
    }

    void apply(Parent root) {
        for (ChatTheme theme : values()) {
            root.getStyleClass().remove(theme.styleClass);
        }
        root.getStyleClass().add(styleClass);
    }

    String styleClass() {
        return styleClass;
    }

    @Override
    public String toString() {
        return label;
    }
}
