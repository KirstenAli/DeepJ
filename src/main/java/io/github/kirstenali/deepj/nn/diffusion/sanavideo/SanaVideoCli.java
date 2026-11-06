package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.function.Supplier;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.Mixin;

/** Video command, available independently and through deepj video. */
@Command(name = "sana-video", mixinStandardHelpOptions = true, description = "Generate a video with SANA-Video 2B")
public final class SanaVideoCli implements Callable<Integer> {
    @Option(names = "--model", required = true) private Path model;
    @Option(names = "--output", defaultValue = "sana-video.mp4") private Path output;
    @Option(names = "--width", defaultValue = "832") private int width;
    @Option(names = "--height", defaultValue = "480") private int height;
    @Option(names = "--frames", defaultValue = "9") private int frames;
    @Option(names = "--steps", defaultValue = "50") private int steps;
    @Option(names = "--guidance", defaultValue = "6") private float guidance;
    @Option(names = "--seed", defaultValue = "42") private long seed;
    @Option(names = "--fps", defaultValue = "16") private int fps;
    @Parameters(arity = "1", description = "Video description") private String prompt;

    private final Supplier<String> backend;

    public SanaVideoCli(Supplier<String> backend) { this.backend = backend; }

    public static CommandLine command() { return new CommandLine(new Standalone()); }

    public static void main(String[] args) { System.exit(command().execute(args)); }

    @Override public Integer call() throws Exception {
        if (!output.toString().toLowerCase(Locale.ROOT).matches(".*\\.(mp4|gif)$") || fps < 1 || fps > 100) {
            throw new IllegalArgumentException("Use a .mp4 or .gif output path and FPS from 1 to 100");
        }
        var options = new SanaVideo.Options(width, height, frames, steps, guidance, seed);
        backend();
        long start = System.nanoTime();
        var images = new SanaVideo(model, System.out::println).generate(prompt, options);
        if (output.toString().toLowerCase(Locale.ROOT).endsWith(".mp4")) VideoMp4.write(images, output, fps);
        else VideoGif.write(images, output, fps);
        System.out.printf("Saved %d frames to %s in %.2f seconds%n", images.size(), output.toAbsolutePath(),
                (System.nanoTime() - start) / 1e9);
        return 0;
    }

    private void backend() {
        String selected = backend.get();
        if (selected.equals("auto")) selected = MetalBackend.isAvailable() ? "metal" : "cpu";
        switch (selected) {
            case "cpu" -> Tensor.setBackend(new CpuBackend());
            case "metal" -> {
                System.setProperty("deepj.metal.maxBufferBytes",
                        System.getProperty("deepj.metal.maxBufferBytes", "6442450944"));
                if (!MetalBackend.isAvailable()) throw new IllegalStateException("Metal backend unavailable");
                Tensor.setBackend(new MetalBackend());
            }
            default -> throw new IllegalArgumentException("Expected cpu or metal backend");
        }
    }

    @Command(name = "sana-video", mixinStandardHelpOptions = true,
            description = "Generate a video with SANA-Video 2B")
    private static final class Standalone implements Callable<Integer> {
        @Option(names = "--backend", defaultValue = "metal") private String backend;
        @Mixin private final SanaVideoCli video = new SanaVideoCli(() -> backend);
        @Override public Integer call() throws Exception { return video.call(); }
    }
}
