package io.github.kirstenali.deepj;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

final class TensorStoragePublicAccessTest {
    @Test
    void inheritedPublicStorageMethodsRemainAccessibleToExternalJavaReflection() throws Exception {
        Tensor tensor = new Tensor(1, 2);
        assertEquals(DType.FLOAT32, Tensor.class.getMethod("dtype").invoke(tensor));
        assertSame(tensor.rawData(), Tensor.class.getMethod("rawData").invoke(tensor));
        assertNull(Tensor.class.getMethod("getGpuTag").invoke(tensor));
        assertFalse((boolean) Tensor.class.getMethod("retainsDeviceBuffer").invoke(tensor));
    }
}
