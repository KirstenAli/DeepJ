package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

class GradientEdgeShapeTest {
    @Test
    void float32GradientsMatchEveryParameterDtypeAndRank() {
        for (DType dtype : DType.values()) {
            for (int rank = 0; rank <= 6; rank++) {
                int[] shape = new int[rank];
                Arrays.fill(shape, 2);
                var edge = new GradientEdge(new Tensor(dtype, shape));
                assertDoesNotThrow(() -> edge.requireShape(new Tensor(shape)));
                assertThrows(IllegalArgumentException.class,
                        () -> edge.requireShape(new Tensor(Arrays.copyOf(shape, shape.length + 1))));
            }
        }
    }

    @Test
    void sameElementCountDoesNotPermitDifferentDimensionsOrRank() {
        var edge = new GradientEdge(new Tensor(2, 3));
        assertThrows(IllegalArgumentException.class, () -> edge.requireShape(new Tensor(3, 2)));
        assertThrows(IllegalArgumentException.class, () -> edge.requireShape(new Tensor(6)));
        assertThrows(NullPointerException.class, () -> edge.requireShape(null));
        assertDoesNotThrow(() -> new GradientEdge(new Tensor(2, 0)).requireShape(new Tensor(2, 0)));
    }

    @Test
    void capturedDimensionsAndAccessorRemainDefensiveSnapshots() {
        int[] shape = {2, 3};
        var edge = new GradientEdge(new Tensor(shape));
        shape[0] = 5;
        edge.shape()[0] = 7;
        assertArrayEquals(new int[]{2, 3}, edge.shape());
        assertDoesNotThrow(() -> edge.requireShape(new Tensor(3, 2).transpose(0, 1)));
        assertThrows(IllegalArgumentException.class, () -> edge.requireShape(new Tensor(shape)));
    }

    @Test
    void shapeValidationDoesNotWeakenAliasMutationDetection() {
        Tensor source = new Tensor(2, 3);
        var edge = new GradientEdge(source);
        source.view(2, 3).set(0, 0, 1);
        assertDoesNotThrow(() -> edge.requireShape(new Tensor(2, 3)));
        assertThrows(IllegalStateException.class, edge::validate);
    }

    @Test
    void independentDetachedCopiesDoNotInvalidateTheOriginalEdge() {
        Tensor source = new Tensor(2, 3);
        var edge = new GradientEdge(source);
        source.detach().set(0, 0, 1);
        assertDoesNotThrow(edge::validate);
        assertEquals(0, source.get(0, 0));
    }

    @Test
    void edgesKeepMetadataButDoNotRetainTheActivationTensorOrStorage() {
        Class<?>[] components = Arrays.stream(GradientEdge.class.getRecordComponents())
                .map(java.lang.reflect.RecordComponent::getType).toArray(Class<?>[]::new);
        assertArrayEquals(new Class<?>[]{GradientState.class, int[].class, long.class}, components);
    }
}
