package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GatherGradientTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void leafGatherReusesTheCanonicalGradientWithoutADenseAdd() {
        List<String> calls = new ArrayList<>();
        Tensor.setBackend(countingBackend(calls));
        Parameter table = new Parameter(Tensor.ones(3, 2));
        Tensor buffer = table.grad();
        Tensor.sliceRows(table.value, new int[]{1, 1, 2}, 2).sumTensor().backward();
        assertSame(buffer, table.grad());
        assertArrayEquals(new float[]{0, 0, 2, 2, 1, 1}, buffer.data);
        assertEquals(1, calls.stream().filter("scatterAddRows"::equals).count());
        assertFalse(calls.contains("addInPlace"));
        assertFalse(calls.contains("addScalar"));
        assertFalse(table.value.gradientState().savedForBackward());
    }

    @Test
    void uninitializedLeafGradientsAccumulateAcrossIndependentPasses() {
        Tensor table = Tensor.ones(3, 2).requiresGrad(true);
        Tensor.sliceRows(table, new int[]{1, 1}, 2).sumTensor().backward();
        Tensor buffer = table.grad();
        Tensor.sliceRows(table, new int[]{1, 2}, 2).sumTensor().backward();
        assertSame(buffer, table.grad());
        assertArrayEquals(new float[]{0, 0, 3, 3, 1, 1}, buffer.data);
        table.zeroGrad();
        Tensor.sliceRows(table, new int[]{0}, 2).sumTensor().backward();
        assertArrayEquals(new float[]{1, 1, 0, 0, 0, 0}, table.grad().data);
    }

    @Test
    void sharedDenseAndGatherBranchesBothReachTheLeaf() {
        Parameter table = new Parameter(Tensor.ones(3, 2));
        Tensor gathered = Tensor.sliceRows(table.value, new int[]{1, 1, 2}, 2);
        table.value.sumTensor().add(gathered.sumTensor()).backward();
        assertArrayEquals(new float[]{1, 1, 3, 3, 2, 2}, table.grad().data);
    }

    @Test
    void multipleGatherBranchesAccumulateWithoutLosingRepeatedRows() {
        Tensor table = Tensor.ones(3, 2).requiresGrad(true);
        Tensor first = Tensor.sliceRows(table, new int[]{1, 1}, 2);
        Tensor second = Tensor.sliceRows(table, new int[]{1, 2}, 2);
        first.sumTensor().add(second.sumTensor()).backward();
        assertArrayEquals(new float[]{0, 0, 3, 3, 1, 1}, table.grad().data);
    }

    @Test
    void nonLeafGatherStillPropagatesThroughItsInputOperation() {
        Tensor table = Tensor.ones(3, 2).requiresGrad(true);
        Tensor intermediate = table.multiplyScalar(2);
        Tensor.sliceRows(intermediate, new int[]{1, 1, 2}, 2).sumTensor().backward();
        assertArrayEquals(new float[]{0, 0, 4, 4, 2, 2}, table.grad().data);
        assertNull(intermediate.grad());
    }

    @Test
    void aliasedUpstreamUsesATemporaryBeforeMutatingTheGradient() {
        Tensor table = Tensor.ones(2, 1).requiresGrad(true);
        Tensor upstream = Tensor.from2D(new float[][]{{1}, {2}});
        table.gradientState().bindGradient(upstream);
        Tensor.sliceRows(table, new int[]{1, 0}, 1).backward(upstream);
        assertArrayEquals(new float[]{3, 3}, table.grad().data);
    }

    @ParameterizedTest
    @CsvSource({"1,2", "3,1"})
    void invalidExistingGradientShapeIsRejected(int rows, int columns) {
        Tensor table = Tensor.ones(3, 2).requiresGrad(true);
        table.gradientState().bindGradient(Tensor.zeros(rows, columns));
        Tensor loss = Tensor.sliceRows(table, new int[]{1}, 2).sumTensor();
        assertThrows(IllegalArgumentException.class, loss::backward);
        assertFalse(table.gradientState().savedForBackward());
    }

    private static TensorBackend countingBackend(List<String> calls) {
        TensorBackend cpu = new CpuBackend();
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, arguments) -> {
                    calls.add(method.getName());
                    return invoke(cpu, method, arguments);
                });
    }

    private static Object invoke(TensorBackend backend, Method method, Object[] arguments) throws Throwable {
        try {
            return method.invoke(backend, arguments);
        } catch (InvocationTargetException exception) {
            throw exception.getCause();
        }
    }
}
