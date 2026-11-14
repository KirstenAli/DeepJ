package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.util.Arrays;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class PackedQ8Test {
    private static final int ROWS = 37;
    private static final int WIDTH = 64;
    private static final int[] HALF_BITS = {0x2400, 0x2800, 0x2000, 0x0123};
    private static final float[] SCALE = {1f / 64, 1f / 32, 1f / 128, Math.scalb(291f, -24)};

    @ParameterizedTest @ValueSource(booleans = {false, true})
    void comparesDecodePrefillAndGatherToDequantizedReference(boolean metal) throws Exception {
        if (metal) assumeTrue(MetalBackend.isAvailable());
        TensorBackend previous = Tensor.backend(), selected = metal ? new MetalBackend() : new CpuBackend();
        Tensor.setBackend(selected);
        try (var weights = read(packedBytes())) {
            Tensor input = Tensor.from(new float[WIDTH], 1, WIDTH);
            assertThrows(IllegalStateException.class, () -> weights.project(input));
            compareOperations(weights);
        } finally {
            selected.releaseResources();
            Tensor.setBackend(previous);
        }
    }

    private static void compareOperations(PackedQ8 weights) {
        try (var ignored = GradMode.noGrad()) {
            for (int batch : new int[]{1, 3, 17, 33}) {
                compareProjection(weights, batch);
            }
            compareGather(weights);
            assertThrows(IllegalArgumentException.class, () -> weights.gather(new int[]{-1}));
            assertThrows(IllegalArgumentException.class, () -> weights.gather(new int[]{ROWS}));
            assertThrows(IllegalArgumentException.class, () -> weights.project(new Tensor(2, 31)));
        }
    }

    @Test void rejectsTruncatedBlocks() {
        byte[] bytes = packedBytes();
        assertThrows(EOFException.class, () -> read(Arrays.copyOf(bytes, bytes.length - 1)));
    }

    private static void compareProjection(PackedQ8 weights, int batch) {
        float[] inputs = new float[(batch + 1) * WIDTH];
        for (int i = 0; i < inputs.length; i++) { inputs[i] = (i % 41 - 19) / 16f; }
        var view = Tensor.from(inputs, batch + 1, WIDTH).slice(0, 1, batch + 1);
        var result = weights.project(view);
        assertTrue(result.hasShape(batch, ROWS));
        float[] values = result.toFloatArray();
        for (int n = 0; n < batch; n++) {
            for (int row = 0; row < ROWS; row++) {
                double sum = 0;
                for (int col = 0; col < WIDTH; col++) {
                    sum += (double) inputs[(n + 1) * WIDTH + col] * value(row, col);
                }
                assertEquals(sum, values[n * ROWS + row], 0.0001);
            }
        }
    }

    private static void compareGather(PackedQ8 weights) {
        int[] indices = {ROWS - 1, 0, 2, 0};
        float[] values = weights.gather(indices).toFloatArray();
        for (int n = 0; n < indices.length; n++) {
            for (int col = 0; col < WIDTH; col++) {
                assertEquals(value(indices[n], col), values[n * WIDTH + col], 0);
            }
        }
    }

    private static float value(int row, int column) {
        return quantized(row, column) * SCALE[(row + column / 32) % SCALE.length];
    }

    private static int quantized(int row, int column) {
        return ((row * 31 + column * 17) % 256) - 128;
    }

    private static byte[] packedBytes() {
        var output = new ByteArrayOutputStream();
        for (int row = 0; row < ROWS; row++) {
            for (int start = 0; start < WIDTH; start += 32) {
                int bits = HALF_BITS[(row + start / 32) % HALF_BITS.length];
                output.write(bits & 255);
                output.write(bits >>> 8);
                for (int col = start; col < start + 32; col++) { output.write(quantized(row, col)); }
            }
        }
        return output.toByteArray();
    }

    private static PackedQ8 read(byte[] bytes) throws Exception {
        return PackedQ8.read(new DataInputStream(new ByteArrayInputStream(bytes)), ROWS, WIDTH);
    }
}
