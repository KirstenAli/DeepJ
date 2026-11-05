package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import java.util.concurrent.Callable;
import javax.imageio.ImageIO;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

/** Standalone image launcher, leaving the existing chat CLI and desktop routing unchanged. */
@Command(name = "sana-image", mixinStandardHelpOptions = true, description = "Generate an image with SANA-Sprint 0.6B")
public final class SanaImageCli implements Callable<Integer> {
    @Option(names = "--model", required = true) private Path model;
    @Option(names = "--output", defaultValue = "sana.png") private Path output;
    @Option(names = "--backend", defaultValue = "metal") private String backend;
    @Option(names = "--width", defaultValue = "1024") private int width;
    @Option(names = "--height", defaultValue = "1024") private int height;
    @Option(names = "--steps", defaultValue = "2") private int steps;
    @Option(names = "--guidance", defaultValue = "4.5") private float guidance;
    @Option(names = "--seed", defaultValue = "42") private long seed;
    @Parameters(arity = "1", description = "Image description") private String prompt;

    public static void main(String[] args) { System.exit(new CommandLine(new SanaImageCli()).execute(args)); }

    @Override public Integer call() throws Exception {
        selectBackend();
        var options = new SanaSprint.Options(width, height, steps, guidance, seed);
        var pipeline = new SanaSprint(model, System.out::println);
        var image = pipeline.generate(prompt, options);
        if (!ImageIO.write(image, "png", output.toFile())) throw new IllegalStateException("PNG writer unavailable");
        System.out.println("Saved " + output.toAbsolutePath());
        return 0;
    }

    private void selectBackend() {
        switch (backend) {
            case "cpu" -> Tensor.setBackend(new CpuBackend());
            case "metal" -> {
                if (!MetalBackend.isAvailable()) throw new IllegalStateException("Metal backend unavailable");
                Tensor.setBackend(new MetalBackend());
            }
            default -> throw new IllegalArgumentException("Backend must be cpu or metal");
        }
    }
}
