package io.github.kirstenali.deepj.chatui;

import javafx.concurrent.Task;
import java.nio.file.Path;

final class VideoGeneration extends Task<DesktopVideo> {
    private final VideoService service;
    private final Path model;
    private final String prompt;
    private final VideoRequest request;
    private DesktopVideo result;

    VideoGeneration(VideoService service, Path model, String prompt, VideoRequest request) {
        this.service = service;
        this.model = model;
        this.prompt = prompt;
        this.request = request;
    }

    @Override protected DesktopVideo call() throws Exception {
        var frames = service.generate(model, prompt, request.options(), this::updateMessage);
        if (isCancelled()) return null;
        result = DesktopVideo.encode(frames, request);
        return result;
    }

    void discard() { if (result != null) result.close(); }
}
