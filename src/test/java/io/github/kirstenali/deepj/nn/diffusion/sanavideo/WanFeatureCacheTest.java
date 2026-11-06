package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.util.Map;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class WanFeatureCacheTest {
    @Test void historyIsBoundedAndIsReleasedBetweenVideos() throws Exception {
        Tensor.setBackend(Boolean.getBoolean("sanavideo.test.metal") ? new MetalBackend() : new CpuBackend());
        try (var ignored = GradMode.noGrad(); var scope = TensorScope.open();
             var weights = TensorWeights.from(Map.of("conv.weight",
                     Tensor.from(new float[]{1, 1, 1}, 1, 1, 3, 1, 1)))) {
            var cache = new WanFeatureCache(weights);
            Tensor last;
            try (cache) {
                for (int frame = 0; frame < 100; frame++) { checkFrame(cache, frame); }
                last = history(cache).get("conv");
            }
            assertTrue(history(cache).isEmpty());
            assertFalse(last.retainsDeviceBuffer());
            try (var fresh = new WanFeatureCache(weights)) { checkFrame(fresh, 0); }
        }
    }

    private static void checkFrame(WanFeatureCache cache, int frame) throws Exception {
        try (var scope = TensorScope.open()) {
            Tensor previous = history(cache).get("conv");
            float value = frame + 1;
            Tensor input = Tensor.from(new float[]{value, value, value, value}, 1, 1, 1, 2, 2);
            float expected = frame == 0 ? 1 : frame == 1 ? 3 : 3 * value - 3;
            assertArrayEquals(new float[]{expected, expected, expected, expected},
                    cache.convolution(input, "conv").toFloatArray(), 0);
            assertEquals(1, history(cache).size());
            assertEquals(4 * Math.min(frame + 1, 2), history(cache).get("conv").numel());
            if (previous != null) assertFalse(previous.retainsDeviceBuffer());
        }
        Tensor.backend().releaseTemporaryResources();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Tensor> history(WanFeatureCache cache) throws Exception {
        var field = WanFeatureCache.class.getDeclaredField("history");
        field.setAccessible(true);
        return (Map<String, Tensor>) field.get(cache);
    }
}
