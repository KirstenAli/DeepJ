package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.Test;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

import static org.junit.jupiter.api.Assertions.*;

class LegacyAttentionBackendTest {
    @Test
    void newOverloadsDelegateEqualHeadsToExistingBackendImplementations() {
        var backend = legacyBackend();
        Tensor input = Tensor.ones(4, 2);
        assertTrue(backend.supportsCausalAttention(input, input, input, 2, 2, 1));
        assertFalse(backend.supportsCausalAttention(input, input, input, 2, 1, 1));
        var saved = backend.causalAttention(input, input, input, 2, 2, 1);
        assertSame(input, saved.output());
        assertSame(input, backend.causalAttentionBackward(input, input, input, saved, input, 2, 2, 1).key());
        assertThrows(UnsupportedOperationException.class,
                () -> backend.causalAttention(input, input, input, 2, 1, 1));
        assertThrows(UnsupportedOperationException.class,
                () -> backend.causalAttentionBackward(input, input, input, saved, input, 2, 1, 1));
    }

    @Test
    void invalidWidthsValuesAndScalesAreRejectedBeforeDispatch() {
        Tensor.setBackend(new CpuBackend());
        Tensor query = new Tensor(8, 2), key = new Tensor(4, 2), wrong = new Tensor(4, 3);
        assertThrows(IllegalArgumentException.class, () -> AttentionOps.causal(query, wrong, wrong, 4, 2, 1));
        assertThrows(IllegalArgumentException.class, () -> AttentionOps.causal(query, key, wrong, 4, 2, 1));
        assertThrows(IllegalArgumentException.class,
                () -> AttentionOps.causal(query, key, key, 4, 2, Float.NaN));
    }

    private static CausalAttentionBackend legacyBackend() {
        return (CausalAttentionBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{CausalAttentionBackend.class}, LegacyAttentionBackendTest::invoke);
    }

    private static Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        if (method.isDefault() && method.getDeclaringClass() == CausalAttentionBackend.class) {
            return InvocationHandler.invokeDefault(proxy, method, args);
        }
        return switch (method.getName()) {
            case "supportsCausalAttention" -> true;
            case "causalAttention" -> new CausalAttentionResult((Tensor) args[0], new Tensor(4, 2));
            case "causalAttentionBackward" -> new AttentionGradients(
                    (Tensor) args[0], (Tensor) args[1], (Tensor) args[2]);
            default -> method.invoke(new CpuBackend(), args);
        };
    }
}
