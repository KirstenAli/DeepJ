package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SanaSamplerTest {
    @Test void completeOneToFourStepSchedulesMatchDiffusers() throws Exception {
        Tensor.setBackend(Boolean.getBoolean("sana.test.metal") ? new MetalBackend() : new CpuBackend());
        try (var fixture = SanaWeights.open(Path.of("src/test/resources/sana"), DType.FLOAT32);
             var ignored = GradMode.noGrad()) {
            for (int count = 1; count <= 4; count++) {
                float[] times = SanaScm.schedule(count);
                for (int step = 0; step < count; step++) { compare(fixture, count, step, times); }
            }
        }
    }

    private static void compare(SanaWeights fixture, int count, int step, float[] times) {
        String name = "scm" + count + "_" + step + ".";
        Tensor latent = fixture.tensor(name + "latent");
        Tensor input = SanaScm.modelInput(latent, SanaScm.time(times[step]));
        assertArrayEquals(fixture.tensor(name + "input").toFloatArray(), input.toFloatArray(), 2e-6f);
        Tensor denoised = SanaScm.denoise(latent, input, fixture.tensor(name + "prediction"), times[step]);
        assertArrayEquals(fixture.tensor(name + "denoised").toFloatArray(), denoised.toFloatArray(), 2e-6f);
        Tensor next = SanaScm.advance(denoised, fixture.tensor(name + "noise"), times[step + 1]);
        assertArrayEquals(fixture.tensor(name + "next").toFloatArray(), next.toFloatArray(), 2e-6f);
    }
}
