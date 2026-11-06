package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideo;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

class VideoWorkerProcessTest {
    private static final Path MODEL = Path.of("src/test/resources/sana-video/pipeline");

    @Test void workerMatchesThePipelineWithoutReplacingTheParentBackend() throws Exception {
        var backend = Boolean.getBoolean("sanavideo.test.metal") ? new MetalBackend() : new CpuBackend();
        Tensor.setBackend(backend);
        var progress = new ArrayList<String>();
        var options = new SanaVideo.Options(16, 16, 5, 1, 1, 4294967338L);
        var frames = new SanaVideoService().generate(MODEL, "A fox", options, progress::add);
        assertEquals(5, frames.size());
        assertEquals(16, frames.get(0).getWidth());
        samePixels(new SanaVideo(MODEL).generate("A fox", options), frames);
        assertTrue(progress.stream().anyMatch(message -> message.contains("denoising step")));
        assertSame(backend, Tensor.backend());
    }

    private static void samePixels(java.util.List<java.awt.image.BufferedImage> expected,
                                   java.util.List<java.awt.image.BufferedImage> actual) {
        for (int index = 0; index < expected.size(); index++) {
            assertArrayEquals(expected.get(index).getRGB(0, 0, 16, 16, null, 0, 16),
                    actual.get(index).getRGB(0, 0, 16, 16, null, 0, 16));
        }
    }

    @Test void workerFailuresReachTheCaller() {
        Tensor.setBackend(new CpuBackend());
        var options = new SanaVideo.Options(16, 16, 5, 1, 1, 42);
        var failure = assertThrows(java.io.IOException.class,
                () -> new SanaVideoService().generate(MODEL.resolve("missing"), "A fox", options, ignored -> {}));
        assertNotNull(failure.getMessage());
    }

    @Test @EnabledIfSystemProperty(named = "sanavideo.test.metal", matches = "true")
    void videoBudgetIsEnforcedEvenWhenTheParentMetalRuntimeAlreadyExists() {
        var backend = new MetalBackend();
        Tensor.setBackend(backend);
        String key = "deepj.metal.maxBufferBytes", previous = System.getProperty(key);
        try {
            System.setProperty(key, "65536");
            var options = new SanaVideo.Options(16, 16, 1, 1, 1, 42);
            var failure = assertThrows(java.io.IOException.class,
                    () -> new SanaVideoService().generate(MODEL, "A fox", options, ignored -> {}));
            assertTrue(failure.getMessage().contains("memory budget"));
            assertSame(backend, Tensor.backend());
        } finally {
            if (previous == null) System.clearProperty(key);
            else System.setProperty(key, previous);
        }
    }

    @Test void cancellationTerminatesTheChildProcess() throws Exception {
        Process process = slowWorker();
        var started = new CountDownLatch(1);
        var stopped = new AtomicBoolean();
        Thread task = new Thread(() -> {
            try { VideoWorkerProcess.await(process, ignored -> started.countDown()); }
            catch (InterruptedException | java.util.concurrent.CancellationException expected) { stopped.set(true); }
            catch (Exception failure) { throw new AssertionError(failure); }
        });
        task.start();
        try {
            assertTrue(started.await(15, TimeUnit.SECONDS));
            task.interrupt();
            task.join(10000);
            assertFalse(task.isAlive());
            assertTrue(stopped.get());
            assertFalse(process.isAlive());
        } finally { process.destroyForcibly(); }
    }

    @Test void interruptedCleanupWaitsForAWorkerThatIgnoresGracefulTermination() throws Exception {
        Process process = slowWorker();
        try (var input = process.inputReader()) {
            assertTrue(input.lines().anyMatch("Encoding video prompt with Gemma 2"::equals));
            Thread.currentThread().interrupt();
            VideoWorkerProcess.stop(process);
            assertFalse(process.isAlive());
            assertTrue(Thread.currentThread().isInterrupted());
        } finally {
            Thread.interrupted();
            process.destroyForcibly();
        }
    }

    private static Process slowWorker() throws Exception {
        String binary = System.getProperty("os.name").startsWith("Windows") ? "java.exe" : "java";
        String java = Path.of(System.getProperty("java.home"), "bin", binary).toString();
        String classes = Path.of(VideoWorkerProcessTest.class.getProtectionDomain().getCodeSource()
                .getLocation().toURI()).toString();
        return new ProcessBuilder(java, "-cp", classes, SlowWorker.class.getName()).redirectErrorStream(true).start();
    }

    public static final class SlowWorker {
        public static void main(String[] arguments) throws Exception {
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try { Thread.sleep(15000); }
                catch (InterruptedException stopped) { Thread.currentThread().interrupt(); }
            }));
            System.out.println("Encoding video prompt with Gemma 2");
            Thread.sleep(30000);
        }
    }
}
