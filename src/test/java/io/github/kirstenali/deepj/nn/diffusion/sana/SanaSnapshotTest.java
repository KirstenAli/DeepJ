package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.nio.file.Path;
import java.util.concurrent.CancellationException;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SanaSnapshotTest {
    private static final Path MODEL = Path.of("src/test/resources/sana/pipeline").toAbsolutePath();

    @Test void pipelineFolderAndStageFilesResolveToTheSameCheckpoint() throws Exception {
        assertEquals(MODEL, SanaSnapshot.directory(MODEL));
        assertEquals(MODEL, SanaSnapshot.directory(MODEL.resolve("model_index.json")));
        assertEquals(MODEL, SanaSnapshot.directory(MODEL.resolve("transformer/config.json")));
        assertTrue(SanaSnapshot.matches(MODEL.resolve("vae/model.safetensors")));
        assertFalse(SanaSnapshot.matches(MODEL.resolve("text_encoder/config.json")));
        assertFalse(SanaSnapshot.matches(Path.of("pom.xml")));
    }

    @Test void cancellationUnwindsGenerationAndRestoresGradientRecording() throws Exception {
        Tensor.setBackend(new CpuBackend());
        var cancelled = new java.util.concurrent.atomic.AtomicBoolean();
        Thread worker = new Thread(() -> {
            Thread.currentThread().interrupt();
            try { new SanaSprint(MODEL).generate("a fox", new SanaSprint.Options(32, 32, 2, 4.5f, 1)); }
            catch (CancellationException expected) {
                cancelled.set(io.github.kirstenali.deepj.autograd.GradMode.isEnabled());
            } catch (java.io.IOException failure) { throw new AssertionError(failure); }
        });
        worker.start();
        worker.join(5000);
        assertFalse(worker.isAlive());
        assertTrue(cancelled.get());
    }
}
