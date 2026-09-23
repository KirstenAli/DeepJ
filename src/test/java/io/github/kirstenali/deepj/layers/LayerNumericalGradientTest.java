package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.layers.transformer.SwiGLULayer;
import io.github.kirstenali.deepj.layers.transformer.attention.MultiHeadLatentAttention;
import io.github.kirstenali.deepj.layers.transformer.attention.MultiHeadSelfAttention;
import io.github.kirstenali.deepj.layers.transformer.attention.RoPEMultiHeadSelfAttention;
import io.github.kirstenali.deepj.layers.transformer.blocks.DeepJOriginTransformerBlock;
import io.github.kirstenali.deepj.layers.transformer.blocks.DeepJPrismTransformerBlock;
import io.github.kirstenali.deepj.layers.transformer.blocks.DeepJOrbitTransformerBlock;
import io.github.kirstenali.deepj.layers.transformer.norm.LayerNorm1D;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertLayerGradients;

class LayerNumericalGradientTest {

    private static final float EPSILON = 1e-3f;
    private static final float ABSOLUTE_TOLERANCE = 2e-3f;
    private static final float RELATIVE_TOLERANCE = 3e-2f;

    @BeforeEach
    void useCpuBackend() {
        Tensor.setBackend(new CpuBackend());
    }

    @Test
    void linearInputAndParameterGradientsMatchFiniteDifferences() {
        check(new Linear(3, 2, new Random(1L)), 2, 3, 2, 11L);
    }

    @Test
    void layerNormInputAndParameterGradientsMatchFiniteDifferences() {
        check(new LayerNorm1D(4), 2, 4, 4, 12L);
    }

    @Test
    void rmsNormInputAndParameterGradientsMatchFiniteDifferences() {
        check(new RMSNorm1D(4), 2, 4, 4, 13L);
    }

    @Test
    void swiGluInputAndParameterGradientsMatchFiniteDifferences() {
        check(new SwiGLULayer(4, 5, new Random(2L)), 2, 4, 4, 14L);
    }

    @Test
    void selfAttentionInputAndParameterGradientsMatchFiniteDifferences() {
        check(new MultiHeadSelfAttention(4, 2, true, new Random(3L)), 3, 4, 4, 15L);
    }

    @Test
    void ropeAttentionInputAndParameterGradientsMatchFiniteDifferences() {
        RotaryEmbedding rope = new RotaryEmbedding(2, 3);
        check(new RoPEMultiHeadSelfAttention(4, 2, true, rope, new Random(4L)), 3, 4, 4, 16L);
    }

    @Test
    void latentAttentionInputAndParameterGradientsMatchFiniteDifferences() {
        RotaryEmbedding rope = new RotaryEmbedding(2, 3);
        check(new MultiHeadLatentAttention(4, 2, 3, 2, rope, new Random(5L)), 3, 4, 4, 17L);
    }

    @Test
    void transformerBlockInputAndParameterGradientsMatchFiniteDifferences() {
        check(new DeepJOriginTransformerBlock(4, 2, 6, new Random(6L)), 3, 4, 4, 18L);
    }

    @Test
    void orbitBlockInputAndParameterGradientsMatchFiniteDifferences() {
        check(new DeepJOrbitTransformerBlock(4, 2, 6, 3, new Random(7L)), 3, 4, 4, 19L);
    }

    @Test
    void prismBlockInputAndParameterGradientsMatchFiniteDifferences() {
        check(new DeepJPrismTransformerBlock(4, 2, 3, 2, 6, 3, new Random(8L)),
                3, 4, 4, 20L);
    }

    private static void check(Layer layer, int rows, int inputCols, int outputCols, long seed) {
        Tensor input = scaledRandom(rows, inputCols, seed);
        Tensor upstream = scaledRandom(rows, outputCols, seed + 100L);
        assertLayerGradients(layer, input, upstream, EPSILON, ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE);
    }

    private static Tensor scaledRandom(int rows, int cols, long seed) {
        return Tensor.random(rows, cols, new Random(seed)).multiplyScalar(0.5f);
    }
}
