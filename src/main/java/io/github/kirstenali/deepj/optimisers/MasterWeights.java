package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Only compact parameters need a separate FLOAT32 master; normal FLOAT32 parameters are never duplicated. */
final class MasterWeights {
    private final Map<Parameter, Entry> values = new IdentityHashMap<>();

    Tensor weights(Parameter parameter) {
        if (parameter.value.dtype() == DType.FLOAT32) return parameter.value;
        Entry entry = values.computeIfAbsent(parameter, key -> new Entry(
                Tensor.backend().cast(key.value, DType.FLOAT32).retainDeviceBuffer(), version(key)));
        if (entry.version() != version(parameter)) {
            throw new IllegalStateException("Compact weights changed outside AdamW; restore or replace the optimizer");
        }
        return entry.weights();
    }

    void assign(Parameter parameter, Tensor master) {
        if (parameter.value == master) return;
        parameter.value.copyFrom(master);
        values.put(parameter, new Entry(master, version(parameter)));
    }

    Map<Integer, Tensor> state(List<Parameter> parameters) {
        Map<Integer, Tensor> result = new LinkedHashMap<>();
        for (int index = 0; index < parameters.size(); index++) {
            Parameter parameter = parameters.get(index);
            if (parameter.value.dtype() != DType.FLOAT32) result.put(index, weights(parameter));
        }
        return result;
    }

    static void validate(List<Parameter> parameters, Map<Integer, Tensor> masters) {
        int count = 0;
        for (int index = 0; index < parameters.size(); index++) {
            Tensor value = parameters.get(index).value;
            if (value.dtype() != DType.FLOAT32) {
                validateMaster(value, masters.get(index));
                count++;
            }
        }
        if (masters.size() != count) throw new IllegalArgumentException("Unexpected AdamW master-weight entries");
    }

    private static void validateMaster(Tensor value, Tensor master) {
        if (master == null || master.dtype() != DType.FLOAT32 || !value.hasSameShape(master)) {
            throw new IllegalArgumentException("Compact parameters require matching FLOAT32 master weights");
        }
    }

    void restore(List<Parameter> parameters, Map<Integer, Tensor> masters) {
        values.clear();
        masters.forEach((index, master) -> {
            Parameter parameter = parameters.get(index);
            values.put(parameter, new Entry(Tensor.backend().copy(master).retainDeviceBuffer(), version(parameter)));
        });
    }

    private static long version(Parameter parameter) { return parameter.value.gradientState().version(); }
    private record Entry(Tensor weights, long version) {}
}
