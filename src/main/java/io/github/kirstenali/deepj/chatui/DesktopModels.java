package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sana.SanaSnapshot;
import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideoSnapshot;
import javafx.concurrent.Task;
import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.function.Consumer;

final class DesktopModels {
    private DesktopModels() {}

    static boolean openMedia(File file, ImageWorkspace images, VideoWorkspace videos) {
        if (SanaVideoSnapshot.matches(file.toPath())) {
            videos.load(file.toPath());
            return true;
        }
        if (SanaSnapshot.matches(file.toPath())) {
            images.load(file.toPath());
            return true;
        }
        return false;
    }

    static void loadChat(File file, ChatService service, ExecutorService worker, Runnable loaded,
                         Consumer<Throwable> failed) {
        Task<Void> task = new Task<>() {
            @Override protected Void call() throws Exception {
                service.loadModel(file.toPath());
                return null;
            }
        };
        task.setOnSucceeded(event -> loaded.run());
        task.setOnFailed(event -> failed.accept(task.getException()));
        worker.execute(task);
    }
}
