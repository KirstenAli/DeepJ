package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import javax.imageio.ImageIO;

/** A separate desktop video process configures its Metal budget before initializing the backend. */
public final class SanaVideoWorker {
    private SanaVideoWorker() {}

    public static void writeRequest(Path path, String prompt, SanaVideo.Options options) throws IOException {
        Files.writeString(path, Qwen3Json.stringify(Map.of("prompt", prompt, "width", options.width(),
                "height", options.height(), "frames", options.frames(), "steps", options.steps(),
                "guidance", options.guidance(), "seed", options.seed())));
    }

    public static void main(String[] arguments) {
        try { generate(arguments); }
        catch (Throwable failure) {
            System.out.println(Qwen3Json.stringify(Map.of("error", message(failure))));
            System.exit(1);
        }
    }

    private static void generate(String[] arguments) throws Exception {
        if (arguments.length != 4) throw new IllegalArgumentException("Expected model, output, request and backend");
        Path directory = Path.of(arguments[1]);
        var request = Qwen3Json.read(Path.of(arguments[2]));
        Tensor.setBackend(arguments[3].equals("metal") ? new MetalBackend() : new CpuBackend());
        var frames = new SanaVideo(Path.of(arguments[0]), System.out::println)
                .generate(Qwen3Json.string(request.get("prompt")), options(request));
        for (int index = 0; index < frames.size(); index++) {
            if (!ImageIO.write(frames.get(index), "png", directory.resolve("frame-" + index + ".png").toFile())) {
                throw new IOException("PNG writer unavailable");
            }
        }
    }

    private static SanaVideo.Options options(Map<String, Object> values) {
        return new SanaVideo.Options(Qwen3Json.integer(values, "width"), Qwen3Json.integer(values, "height"),
                Qwen3Json.integer(values, "frames"), Qwen3Json.integer(values, "steps"),
                ((Number) values.get("guidance")).floatValue(), Qwen3Json.integer(values.get("seed")));
    }

    private static String message(Throwable failure) {
        String message = failure.getMessage();
        return message == null || message.isBlank() ? failure.getClass().getSimpleName() : message;
    }
}
