package io.github.kirstenali.deepj.chatui;

import javafx.stage.FileChooser;
import javafx.stage.Stage;
import java.io.File;

final class DesktopModelPicker {
    private DesktopModelPicker() {}

    static File choose(Stage stage) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Choose a local model");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("DeepJ and Hugging Face models",
                "*.dj", "*.bin", "config.json", "model_index.json", "*.safetensors", "*.safetensors.index.json"));
        return chooser.showOpenDialog(stage);
    }
}
