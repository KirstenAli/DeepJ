package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.autograd.GradMode;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.HashMap;
import java.util.Map;

/** Test-only reconstruction of the previous expanded-head fused attention path. */
final class ExpandedAttentionReferenceBackend {
    private final MetalBackend metal;
    private final Map<CausalAttentionResult, Saved> pending = new HashMap<>();

    private ExpandedAttentionReferenceBackend(MetalBackend metal) { this.metal = metal; }

    static TensorBackend create(MetalBackend metal) {
        var reference = new ExpandedAttentionReferenceBackend(metal);
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{CausalAttentionBackend.class}, reference::invoke);
    }

    private Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        if (method.getName().equals("causalAttention") && args.length == 6) return forward(args);
        if (method.getName().equals("causalAttentionBackward") && args.length == 8) return backward(args);
        try { return method.invoke(metal, args); }
        catch (InvocationTargetException error) { throw error.getCause(); }
    }

    private CausalAttentionResult forward(Object[] args) {
        Tensor query = (Tensor) args[0], key = (Tensor) args[1], value = (Tensor) args[2];
        int queryHeads = (int) args[3], keyHeads = (int) args[4];
        int[] mapping = mapping(query.matrixRows(), queryHeads, keyHeads);
        Tensor expandedKey = metal.sliceRows(key, mapping), expandedValue = metal.sliceRows(value, mapping);
        var saved = metal.causalAttention(query, expandedKey, expandedValue, queryHeads, (float) args[5]);
        if (GradMode.isEnabled()) pending.put(saved, new Saved(expandedKey, expandedValue, mapping));
        return saved;
    }

    private AttentionGradients backward(Object[] args) {
        Tensor query = (Tensor) args[0], key = (Tensor) args[1], value = (Tensor) args[2];
        var saved = (CausalAttentionResult) args[3];
        Saved state = pending.remove(saved);
        if (state == null) throw new IllegalStateException("Missing expanded attention reference state");
        var gradients = metal.causalAttentionBackward(query, state.key(), state.value(), saved,
                (Tensor) args[4], (int) args[5], (float) args[7]);
        return new AttentionGradients(gradients.query(), fold(key, state.mapping(), gradients.key()),
                fold(value, state.mapping(), gradients.value()));
    }

    private Tensor fold(Tensor compact, int[] mapping, Tensor gradient) {
        Tensor result = new Tensor(compact.matrixRows(), compact.matrixCols());
        metal.scatterAddRows(result, mapping, gradient);
        return result;
    }

    private static int[] mapping(int rows, int queryHeads, int keyHeads) {
        int length = rows / queryHeads, groupSize = queryHeads / keyHeads;
        int[] mapping = new int[rows];
        for (int row = 0; row < rows; row++) {
            mapping[row] = (row / length / groupSize) * length + row % length;
        }
        return mapping;
    }

    private record Saved(Tensor key, Tensor value, int[] mapping) {}
}
