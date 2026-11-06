package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideo;
import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideoWorker;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import javax.imageio.ImageIO;

/** Owns a bounded video JVM and its temporary frame exchange; cancellation also terminates the worker. */
final class VideoWorkerProcess {
    private VideoWorkerProcess() {}

    static List<BufferedImage> generate(Path model, String prompt, SanaVideo.Options options, boolean metal,
                                         Consumer<String> progress) throws Exception {
        ModelMemory.checkCancelled();
        Path directory = Files.createTempDirectory("deepj-video-worker-");
        try {
            Path request = directory.resolve("request.json");
            SanaVideoWorker.writeRequest(request, prompt, options);
            Process process = new ProcessBuilder(command(model, directory, request, metal))
                    .redirectErrorStream(true).start();
            await(process, progress);
            return frames(directory, options.frames());
        } finally { clean(directory); }
    }

    private static List<String> command(Path model, Path directory, Path request, boolean metal) throws Exception {
        String binary = System.getProperty("os.name").startsWith("Windows") ? "java.exe" : "java";
        String executable = Path.of(System.getProperty("java.home"), "bin", binary).toString();
        String classes = Path.of(SanaVideoWorker.class.getProtectionDomain().getCodeSource().getLocation().toURI())
                .toString();
        long heap = metal ? 2560L * 1024 * 1024 : Runtime.getRuntime().maxMemory();
        var command = new ArrayList<>(List.of(executable, "-Xmx" + heap, "--enable-native-access=ALL-UNNAMED",
                "-Djava.awt.headless=true",
                "-Ddeepj.metal.maxBufferBytes=" + System.getProperty("deepj.metal.maxBufferBytes", "6442450944")));
        for (String property : List.of("deepj.video.weightCacheBytes", "java.library.path")) {
            if (System.getProperty(property) != null) command.add("-D" + property + "=" + System.getProperty(property));
        }
        command.addAll(List.of("-cp", classes, SanaVideoWorker.class.getName(), model.toAbsolutePath().toString(),
                directory.toString(), request.toString(), metal ? "metal" : "cpu"));
        return command;
    }

    static void await(Process process, Consumer<String> progress) throws Exception {
        var failure = new AtomicReference<String>();
        Thread output = reader(process, progress, failure);
        try {
            while (!process.waitFor(200, TimeUnit.MILLISECONDS)) {
                ModelMemory.checkCancelled();
                if (failure.get() != null) throw new IOException(failure.get());
            }
            output.join(2000);
            if (process.exitValue() != 0) {
                throw new IOException(failure.get() == null
                        ? "The video worker stopped before finishing." : failure.get());
            }
        } finally {
            stop(process);
            output.interrupt();
        }
    }

    private static Thread reader(Process process, Consumer<String> progress, AtomicReference<String> failure) {
        Thread thread = new Thread(() -> {
            try (var input = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                for (String line; (line = input.readLine()) != null;) { report(line, progress, failure); }
            } catch (IOException | RuntimeException error) {
                if (process.isAlive()) failure.set("Could not read video progress: " + error.getMessage());
            }
        }, "deepj-video-progress");
        thread.setDaemon(true);
        thread.start();
        return thread;
    }

    private static void report(String line, Consumer<String> progress, AtomicReference<String> failure) {
        if (line.startsWith("{\"error\":")) {
            String message = Qwen3Json.string(Qwen3Json.object(Qwen3Json.parse(line)).get("error"));
            failure.set(message.contains("Metal allocation budget exceeded")
                    ? "This video exceeds the memory budget. Try fewer frames or a smaller resolution." : message);
        } else if (line.startsWith("Encoding ") || line.startsWith("Loading ")
                || line.startsWith("Video denoising ") || line.startsWith("Decoding ")) progress.accept(line);
    }

    private static List<BufferedImage> frames(Path directory, int count) throws IOException {
        var frames = new ArrayList<BufferedImage>();
        for (int index = 0; index < count; index++) {
            BufferedImage image = ImageIO.read(directory.resolve("frame-" + index + ".png").toFile());
            if (image == null) throw new IOException("The video worker produced an unreadable frame.");
            frames.add(image);
        }
        return frames;
    }

    static void stop(Process process) {
        if (!process.isAlive()) return;
        var children = children(process);
        process.destroy();
        children.forEach(ProcessHandle::destroy);
        waitStopped(process);
        if (process.isAlive()) {
            children.forEach(ProcessHandle::destroyForcibly);
            process.destroyForcibly();
            waitStopped(process);
        }
    }

    private static void waitStopped(Process process) {
        boolean interrupted = Thread.interrupted();
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        try {
            while (process.isAlive() && System.nanoTime() < deadline) {
                try { process.waitFor(Math.max(1, deadline - System.nanoTime()), TimeUnit.NANOSECONDS); }
                catch (InterruptedException stopped) { interrupted = true; }
            }
        } finally {
            if (interrupted) Thread.currentThread().interrupt();
        }
    }

    private static List<ProcessHandle> children(Process process) {
        try { return process.descendants().toList(); }
        catch (RuntimeException restricted) {
            // Process enumeration can be restricted independently of permission to stop our own child.
            return List.of();
        }
    }

    private static void clean(Path directory) throws IOException {
        try (var files = Files.list(directory)) {
            for (Path file : files.toList()) { Files.deleteIfExists(file); }
        }
        Files.deleteIfExists(directory);
    }
}
