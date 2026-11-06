package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.tensor.Tensor;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Video jobs share the desktop model worker; preview files and downloads have a separate I/O worker. */
public final class VideoWorkspace {
    private final Stage stage;
    private final ExecutorService worker;
    private final ChatService chat;
    private final ImageWorkspace.Events events;
    private final VideoService service;
    private final VideoDownloads downloads;
    private final Map<Integer, DesktopVideo> videos = new HashMap<>();
    private final ExecutorService files = Executors.newSingleThreadExecutor(action -> {
        var thread = new Thread(action, "deepj-video-files");
        thread.setDaemon(true);
        return thread;
    });
    private VideoGeneration active;
    private Path model;
    private int sequence, response;
    private boolean closed;

    VideoWorkspace(Stage stage, ExecutorService worker, ChatService chat, ImageWorkspace.Events events) {
        this(stage, worker, chat, events, new SanaVideoService(), new VideoDownloads());
    }

    VideoWorkspace(Stage stage, ExecutorService worker, ChatService chat, ImageWorkspace.Events events,
                    VideoService service, VideoDownloads downloads) {
        this.stage = stage;
        this.worker = worker;
        this.chat = chat;
        this.events = events;
        this.service = service;
        this.downloads = downloads;
        stage.addEventHandler(WindowEvent.WINDOW_HIDDEN, event -> close());
    }

    public void chooseModel() {
        if (events.busy().getAsBoolean()) return;
        var picker = new DirectoryChooser();
        picker.setTitle("Choose a SANA-Video model folder");
        var selection = picker.showDialog(stage);
        if (selection != null) load(selection.toPath());
    }

    void load(Path path) {
        if (events.busy().getAsBoolean()) return;
        events.busyChanged().accept(true);
        show("setStatus", "Opening video model…");
        Task<Path> task = resolve(path);
        task.setOnSucceeded(event -> opened(task.getValue()));
        task.setOnFailed(event -> {
            show("setStatus", "Could not open video model: " + DesktopErrors.message(task.getException()));
            events.busyChanged().accept(false);
        });
        worker.execute(task);
    }

    private Task<Path> resolve(Path path) {
        return new Task<>() {
            @Override protected Path call() throws Exception {
                Path selected = service.resolve(path);
                chat.unloadModel();
                Tensor.backend().releaseResources();
                ChatBackend.configure();
                return selected;
            }
        };
    }

    private void opened(Path path) {
        model = path;
        events.modelChanged().run();
        show("setAgentWorkspace", "");
        show("setVideoModel", "SANA-Video 2B");
        show("setStatus", ChatBackend.readyStatus());
        events.busyChanged().accept(false);
    }

    public void generate(String prompt, String settings) {
        if (model == null || events.busy().getAsBoolean() || prompt == null || prompt.isBlank()) return;
        try { start(prompt.strip(), VideoRequest.read(settings)); }
        catch (IllegalArgumentException | ArithmeticException failure) { show("toast", failure.getMessage()); }
    }

    private void start(String prompt, VideoRequest request) {
        int id = sequence++;
        var task = new VideoGeneration(service, model, prompt, request);
        active = task;
        response = id;
        show("startVideo", prompt, id);
        events.busyChanged().accept(true);
        task.messageProperty().addListener((ob, old, text) -> {
            if (task == active && !task.isCancelled()) show("videoProgress", id, text);
        });
        worker.execute(() -> {
            task.run();
            Platform.runLater(() -> finished(task, id));
        });
    }

    private void finished(VideoGeneration task, int id) {
        if (closed) {
            task.discard();
            return;
        }
        if (task != active) return;
        if (task.isCancelled() || response != id) {
            task.discard();
            show("cancelVideo", id);
        } else if (task.getException() != null) show("failVideo", id, DesktopErrors.message(task.getException()));
        else completed(id, task.getValue());
        active = null;
        show("setGenerating", false);
        show("setStatus", ChatBackend.readyStatus());
        events.busyChanged().accept(false);
    }

    private void completed(int id, DesktopVideo video) {
        videos.put(id, video);
        var options = video.request().options();
        show("finishVideo", id, video.frameUrls(),
                options.width(), options.height(), video.request().fps());
    }

    public void stop() {
        if (active != null) {
            active.cancel(true);
            show("stoppingVideo", response);
        }
    }

    public void save(int id) {
        DesktopVideo video = videos.get(id);
        if (video != null) downloads.choose(stage, video, files, text -> show("toast", text));
    }

    void clear() {
        videos.values().forEach(video -> files.execute(video::close));
        videos.clear();
        response = -1;
    }

    void leave() { model = null; }

    private void close() {
        show("clearVideoPlayers");
        closed = true;
        if (active != null) active.cancel(true);
        clear();
        files.shutdown();
    }

    private void show(String method, Object... values) { if (!closed) events.ui().call(method, values); }
}
