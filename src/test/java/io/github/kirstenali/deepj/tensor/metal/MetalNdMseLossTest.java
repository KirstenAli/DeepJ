package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.loss.NdMseLossContract;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalNdMseLossTest extends NdMseLossContract {
    private TensorBackend previous;
    private MetalBackend metal;

    @BeforeEach void metal() {
        assumeTrue(MetalBackend.isAvailable(), "Metal device not available");
        previous = Tensor.backend();
        metal = new MetalBackend();
        Tensor.setBackend(metal);
    }

    @AfterEach void restore() {
        if (metal != null) metal.releaseResources();
        if (previous != null) Tensor.setBackend(previous);
    }

    @ParameterizedTest @EnumSource(DType.class)
    void dtypeCastsAndLossKeepDeviceStorageLazy(DType dtype) {
        Tensor predicted = metal.copy(Tensor.ones(dtype, 2, 3, 4)).requiresGrad(true);
        Tensor target = metal.copy(Tensor.zeros(dtype, 2, 3, 4));
        var mse = new MSELoss();
        Tensor loss = mse.forward(predicted, target), gradient = mse.gradient(predicted, target);
        metal.graph.flush();
        assertFalse(TensorTestStorage.hasHostStorage(predicted));
        assertFalse(TensorTestStorage.hasHostStorage(target));
        assertFalse(TensorTestStorage.hasHostStorage(loss));
        assertFalse(TensorTestStorage.hasHostStorage(gradient));
        assertEquals(24L * dtype.bytes(), predicted.storageBytes());
        assertEquals(DType.FLOAT32, gradient.dtype());
        loss.backward();
        assertEquals(1, loss.item());
        assertArrayEquals(gradient.toFloatArray(), predicted.grad().toFloatArray());
    }
}
