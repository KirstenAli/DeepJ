package io.github.kirstenali.deepj.chatui;

import javafx.concurrent.Task;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.ExecutorService;
import java.util.function.BiFunction;
import java.util.function.Consumer;

final class VideoDownloads {
    private final BiFunction<FileChooser, Stage, File> picker;
    VideoDownloads() { this(FileChooser::showSaveDialog); }
    VideoDownloads(BiFunction<FileChooser, Stage, File> picker) { this.picker = picker; }

    void choose(Stage owner, DesktopVideo video, ExecutorService worker, Consumer<String> status) {
        var chooser = new FileChooser();
        chooser.setTitle("Save video");
        chooser.setInitialFileName(video.filename());
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("MP4 video", "*.mp4"));
        File destination = picker.apply(chooser, owner);
        if (destination == null) return;
        Task<Void> task = copy(video, destination);
        task.setOnSucceeded(event -> status.accept("Video saved"));
        task.setOnFailed(event -> status.accept("Could not save video: " + DesktopErrors.message(task.getException())));
        worker.execute(task);
    }

    private static Task<Void> copy(DesktopVideo video, File destination) {
        return new Task<>() {
            @Override protected Void call() throws Exception {
                Files.copy(video.mp4(), destination.toPath(), StandardCopyOption.REPLACE_EXISTING);
                return null;
            }
        };
    }
}
