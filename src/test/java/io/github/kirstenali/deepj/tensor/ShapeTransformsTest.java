package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

final class ShapeTransformsTest {
    @Test
    void broadcastingAlignsTrailingAxesAndKeepsZeroSizeAxesEmpty() {
        assertEquals(new TensorShape(5, 3, 4, 1),
                ShapeTransforms.broadcast(new TensorShape(5, 1, 4, 1), new TensorShape(3, 1, 1)));
        assertEquals(new TensorShape(2, 0, 3),
                ShapeTransforms.broadcast(new TensorShape(1, 0, 1), new TensorShape(2, 1, 3)));
        assertEquals(new TensorShape(2, 3), ShapeTransforms.broadcast(new TensorShape(), new TensorShape(2, 3)));
        assertEquals(new TensorShape(), ShapeTransforms.broadcast(new TensorShape(), new TensorShape()));
        assertThrows(IllegalArgumentException.class,
                () -> ShapeTransforms.broadcast(new TensorShape(2, 3), new TensorShape(2)));
    }

    @Test
    void reshapeCanInferOneAxisWithoutMutatingTheRequest() {
        var source = new TensorShape(2, 3, 4);
        int[] request = {3, -1};
        assertEquals(new TensorShape(3, 8), ShapeTransforms.reshape(source, request));
        assertArrayEquals(new int[]{3, -1}, request);
        assertEquals(new TensorShape(24), ShapeTransforms.reshape(source, -1));
        assertEquals(new TensorShape(), ShapeTransforms.reshape(new TensorShape(1, 1)));
        assertEquals(new TensorShape(0, 3), ShapeTransforms.reshape(new TensorShape(0), -1, 3));
        assertEquals(new TensorShape(2, 0), ShapeTransforms.reshape(new TensorShape(0), 2, 0));
    }

    @Test
    void reshapeRejectsAmbiguityInvalidDimensionsAndChangedCounts() {
        var source = new TensorShape(2, 3);
        assertThrows(IllegalArgumentException.class, () -> ShapeTransforms.reshape(source, -1, -1));
        assertThrows(IllegalArgumentException.class, () -> ShapeTransforms.reshape(source, -2));
        assertThrows(IllegalArgumentException.class, () -> ShapeTransforms.reshape(source, 4));
        assertThrows(IllegalArgumentException.class, () -> ShapeTransforms.reshape(source, 4, -1));
        assertThrows(IllegalArgumentException.class,
                () -> ShapeTransforms.reshape(new TensorShape(0), 0, -1));
        assertThrows(NullPointerException.class, () -> ShapeTransforms.reshape(source, (int[]) null));
    }
}
