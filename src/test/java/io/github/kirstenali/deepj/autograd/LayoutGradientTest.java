package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class LayoutGradientTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void batchedRectangularProductsDifferentiateBothTransposeFlags(boolean ta, boolean tb) {
        Tensor a = point(2 * (ta ? 3 : 2), ta ? 2 : 3);
        Tensor b = point(2 * (tb ? 4 : 3), tb ? 3 : 4);
        OperationGradientAssertions.check(x -> LayoutOps.batchedMatmul(x, b, 2, ta, tb), a);
        OperationGradientAssertions.check(x -> LayoutOps.batchedMatmul(a, x, 2, ta, tb), b);
    }

    @Test
    void headSplittingAndMergingPreserveGradientLayout() {
        OperationGradientAssertions.check(x -> LayoutOps.splitHeads(x, 2).pow(2), point(3, 8));
        OperationGradientAssertions.check(x -> LayoutOps.mergeHeads(x, 2).pow(2), point(6, 4));
    }

    @ParameterizedTest
    @CsvSource({"false", "true"})
    void rotaryUsesTheOppositeRotationForItsDerivative(boolean inverse) {
        Tensor cosine = point(3, 2), sine = point(3, 2).multiplyScalar(0.3f);
        OperationGradientAssertions.check(x -> LayoutOps.rotary(x, cosine, sine, 3, inverse), point(6, 4));
    }

    @Test
    void rotaryTablesAreExplicitlyConstant() {
        Tensor input = point(2, 4), cosine = point(2, 2), sine = point(2, 2);
        cosine.requiresGrad(true);
        assertThrows(IllegalArgumentException.class, () -> LayoutOps.rotary(input, cosine, sine, 2, false));
        cosine.requiresGrad(false);
        sine.requiresGrad(true);
        assertThrows(IllegalArgumentException.class, () -> LayoutOps.rotary(input, cosine, sine, 2, false));
    }

    @Test
    void gatheringSnapshotsIndicesAndAddsDuplicateRows() {
        Tensor table = Tensor.ones(3, 2).requiresGrad(true);
        int[] indices = {1, 1, 2};
        Tensor output = Tensor.sliceRows(table, indices, 2);
        indices[0] = 0;
        output.sumTensor().backward();
        assertArrayEquals(new float[]{0, 0, 2, 2, 1, 1}, table.grad().data);
    }

    @Test
    void randomRowSamplingRemainsDifferentiableAndReproducible() {
        Tensor table = Tensor.ones(3, 2).requiresGrad(true);
        Tensor.sampleRows(table, 5, new Random(42)).sumTensor().backward();
        int[] counts = new int[3];
        Random random = new Random(42);
        for (int index = 0; index < 5; index++) {
            counts[random.nextInt(3)]++;
        }
        assertArrayEquals(new float[]{counts[0], counts[0], counts[1], counts[1], counts[2], counts[2]},
                table.grad().data);
    }

    private static Tensor point(int rows, int columns) {
        return Tensor.random(rows, columns, new Random(117)).addScalar(0.4f);
    }
}
