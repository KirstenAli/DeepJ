package io.github.kirstenali.deepj.chatui;

import javafx.concurrent.Task;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import java.io.File;
import java.nio.file.Files;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/** Saves encoded PNG bytes on an independent I/O worker, including during image generation. */
final class ImageDownloads implements AutoCloseable {
    private final java.util.function.BiFunction<FileChooser, Stage, File> picker;
    private final ExecutorService worker = Executors.newSingleThreadExecutor(runnable -> {
        var thread = new Thread(runnable, "deepj-image-save");
        thread.setDaemon(true);
        return thread;
    });

    ImageDownloads() { this(FileChooser::showSaveDialog); }

    ImageDownloads(java.util.function.BiFunction<FileChooser, Stage, File> picker) { this.picker = picker; }

    void choose(Stage owner, DesktopImage image, Consumer<String> status) {
        var chooser = new FileChooser();
        chooser.setTitle("Save image");
        chooser.setInitialFileName(image.filename());
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("PNG image", "*.png"));
        File destination = picker.apply(chooser, owner);
        if (destination != null) save(destination, image, status);
    }

    void save(File destination, DesktopImage image, Consumer<String> status) {
        Task<Void> task = new Task<>() {
            @Override protected Void call() throws Exception {
                Files.write(destination.toPath(), image.png());
                return null;
            }
        };
        task.setOnSucceeded(event -> status.accept("Image saved"));
        task.setOnFailed(event -> status.accept("Could not save image: " + task.getException().getMessage()));
        worker.execute(task);
    }

    @Override public void close() { worker.shutdown(); }
}
