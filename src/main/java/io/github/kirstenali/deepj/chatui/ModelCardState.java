package io.github.kirstenali.deepj.chatui;

import javafx.scene.control.Label;

final class ModelCardState {

    private static final String[] STYLES = {
        "model-idle", "model-loading", "model-ready", "model-error"
    };

    private ModelCardState() {
    }

    static void show(Label nameLabel, Label detailsLabel, Label statusDot,
                     String name, String details, String style) {
        nameLabel.setText(name);
        detailsLabel.setText(details);
        statusDot.getStyleClass().removeAll(STYLES);
        statusDot.getStyleClass().add(style);
    }
}
