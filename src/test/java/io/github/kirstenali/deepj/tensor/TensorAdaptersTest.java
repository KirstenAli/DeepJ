package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorAdaptersTest {

    // -- packF32 -------------------------------------------------------------

    @Test
    void packF32ProducesRowMajorFloatArray() {
        Tensor t = Tensor.from2D(new double[][]{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        });

        float[] packed = TensorAdapters.packF32(t);

        assertEquals(6, packed.length);
        assertArrayEquals(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, packed);
    }

    @Test
    void packF32SingleElement() {
        Tensor t = Tensor.from2D(new double[][]{{42.5}});
        float[] packed = TensorAdapters.packF32(t);
        assertEquals(1, packed.length);
        assertEquals(42.5f, packed[0]);
    }

    @Test
    void packF32TruncatesDoublePrecision() {
        // 1/3 has different representations in double vs float
        Tensor t = Tensor.from2D(new double[][]{{1.0 / 3.0}});
        float[] packed = TensorAdapters.packF32(t);
        assertEquals(1.0f / 3.0f, packed[0]);
    }

    // -- unpackF32 -----------------------------------------------------------

    @Test
    void unpackF32CreatesCorrectTensor() {
        float[] flat = {1f, 2f, 3f, 4f, 5f, 6f};
        Tensor t = TensorAdapters.unpackF32(flat, 2, 3);

        assertEquals(2, t.rows);
        assertEquals(3, t.cols);
        assertEquals(1.0, t.data[0 * 3 + 0], 1e-6);
        assertEquals(3.0, t.data[0 * 3 + 2], 1e-6);
        assertEquals(4.0, t.data[1 * 3 + 0], 1e-6);
        assertEquals(6.0, t.data[1 * 3 + 2], 1e-6);
    }

    @Test
    void unpackF32RejectsLengthMismatch() {
        float[] flat = {1f, 2f, 3f};
        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> TensorAdapters.unpackF32(flat, 2, 3)
        );
        assertTrue(ex.getMessage().contains("does not match shape"));
    }

    // -- unpackF32Into -------------------------------------------------------

    @Test
    void unpackF32IntoOverwritesExistingData() {
        Tensor t = Tensor.from2D(new double[][]{
                {99, 99, 99},
                {99, 99, 99}
        });

        float[] flat = {10f, 20f, 30f, 40f, 50f, 60f};
        TensorAdapters.unpackF32Into(flat, t);

        assertEquals(10.0, t.data[0 * 3 + 0], 1e-6);
        assertEquals(30.0, t.data[0 * 3 + 2], 1e-6);
        assertEquals(60.0, t.data[1 * 3 + 2], 1e-6);
    }

    // -- round-trip ----------------------------------------------------------

    @Test
    void packThenUnpackIsIdentity() {
        Tensor original = Tensor.from2D(new double[][]{
                {1.5, -2.5},
                {0.0,  3.0},
                {7.0, -1.0}
        });

        float[] packed = TensorAdapters.packF32(original);
        Tensor restored = TensorAdapters.unpackF32(packed, 3, 2);

        assertEquals(original.rows, restored.rows);
        assertEquals(original.cols, restored.cols);
        for (int r = 0; r < original.rows; r++) {
            for (int c = 0; c < original.cols; c++) {
                // float precision: compare at float tolerance
                assertEquals(original.data[r * original.cols + c], restored.data[r * restored.cols + c]);
            }
        }
    }
}
