package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sana.SanaSprint;
import io.github.kirstenali.deepj.tensor.Tensor;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import java.io.File;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/** Image bridge for the existing desktop window; chat and image generation share one model worker. */
public final class ImageWorkspace {
    @FunctionalInterface interface Ui { void call(String name, Object... arguments); }
    record Events(BooleanSupplier busy, Consumer<Boolean> busyChanged, Runnable modelChanged, Ui ui) {}
    private final Stage stage;
    private final ExecutorService worker;
    private final ChatService chat;
    private final Events events;
    private final ImageService service;
    private final ImageDownloads downloads;
    private final Map<Integer, DesktopImage> images = new HashMap<>();
    private Path model;
    private String chatTitle;
    private Task<DesktopImage> active;
    private int response, sequence;
    private boolean closed;

    ImageWorkspace(Stage stage, ExecutorService worker, ChatService chat, Events events) {
        this(stage, worker, chat, events, new SanaImageService());
    }

    ImageWorkspace(Stage stage, ExecutorService worker, ChatService chat, Events events, ImageService service) {
        this(stage, worker, chat, events, service, new ImageDownloads());
    }

    ImageWorkspace(Stage stage, ExecutorService worker, ChatService chat, Events events, ImageService service,
                   ImageDownloads downloads) {
        this.downloads = downloads;
        this.stage = stage;
        this.worker = worker;
        this.chat = chat;
        this.events = events;
        this.service = service;
        stage.addEventHandler(WindowEvent.WINDOW_HIDDEN, event -> close());
    }

    public void chooseModel() {
        if (events.busy().getAsBoolean()) return;
        var chooser = new DirectoryChooser();
        chooser.setTitle("Choose a SANA-Sprint model folder");
        if (model != null) chooser.setInitialDirectory(model.toFile());
        File chosen = chooser.showDialog(stage);
        if (chosen != null) load(chosen.toPath());
    }

    void load(Path selection) {
        if (events.busy().getAsBoolean()) return;
        events.busyChanged().accept(true);
        show("setStatus", "Opening image model…");
        Task<Path> task = new Task<>() {
            @Override protected Path call() throws Exception {
                Path selected = service.resolve(selection);
                chat.unloadModel();
                Tensor.backend().releaseResources();
                ChatBackend.configure();
                return selected;
            }
        };
        task.setOnSucceeded(event -> loaded(task.getValue()));
        task.setOnFailed(event -> loadFailed(task.getException()));
        worker.execute(task);
    }

    private void loaded(Path selected) {
        if (model == null) chatTitle = stage.getTitle();
        model = selected;
        stage.setTitle("DeepJ Images");
        events.modelChanged().run();
        show("setAgentWorkspace", "");
        show("setImageModel", "SANA-Sprint 0.6B");
        show("setStatus", ChatBackend.readyStatus());
        events.busyChanged().accept(false);
    }

    private void loadFailed(Throwable error) {
        show("setStatus", "Could not open image model: " + message(error));
        events.busyChanged().accept(false);
    }

    public void generate(String prompt, int width, int height, int steps, long seed, double guidance) {
        if (model == null || events.busy().getAsBoolean() || prompt == null || prompt.isBlank()) return;
        try {
            start(prompt.strip(), new SanaSprint.Options(width, height, steps, (float) guidance, seed));
        } catch (IllegalArgumentException error) {
            show("toast", error.getMessage());
        }
    }

    private void start(String prompt, SanaSprint.Options options) {
        Path selected = model;
        int id = sequence++;
        Task<DesktopImage> task = new Task<>() {
            @Override protected DesktopImage call() throws Exception {
                return DesktopImage.encode(service.generate(selected, prompt, options, this::updateMessage), options);
            }
        };
        active = task;
        response = id;
        events.busyChanged().accept(true);
        show("startImage", prompt, id);
        task.messageProperty().addListener((observable, before, text) -> progress(task, id, text));
        worker.execute(() -> {
            task.run();
            Platform.runLater(() -> finish(task, id));
        });
    }

    private void progress(Task<DesktopImage> task, int id, String text) {
        if (task == active && !task.isCancelled()) show("imageProgress", id, text);
    }

    private void finish(Task<DesktopImage> task, int id) {
        if (closed || task != active) return;
        if (response == id) showResult(task, id);
        active = null;
        show("setGenerating", false);
        show("setStatus", ChatBackend.readyStatus());
        events.busyChanged().accept(false);
    }

    private void showResult(Task<DesktopImage> task, int id) {
        if (task.isCancelled()) show("cancelImage", id);
        else if (task.getException() != null) show("failImage", id, message(task.getException()));
        else complete(id, task.getValue());
    }

    private void complete(int id, DesktopImage image) {
        images.put(id, image);
        show("finishImage", id, image.dataUrl(), image.width(), image.height(), image.steps(), image.seed());
    }

    public void stop() {
        if (active == null) return;
        active.cancel(true);
        show("stoppingImage", response);
    }

    public void save(int id) {
        DesktopImage image = images.get(id);
        if (image != null) downloads.choose(stage, image, message -> show("toast", message));
    }

    void clear() {
        images.clear();
        if (active != null) response = -1;
    }
    void leave() {
        model = null;
        if (chatTitle != null) stage.setTitle(chatTitle);
    }

    private void close() {
        closed = true;
        if (active != null) active.cancel(true);
        images.clear();
        downloads.close();
    }

    private void show(String name, Object... arguments) { if (!closed) events.ui().call(name, arguments); }
    private static String message(Throwable error) {
        return error == null || error.getMessage() == null ? "Unknown error" : error.getMessage();
    }
}
