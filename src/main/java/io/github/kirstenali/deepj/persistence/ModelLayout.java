package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.TensorDescriptor;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/** Validated model construction metadata,without allocating or converting tensor values. */
public record ModelLayout(ModelSettings settings, Map<String, TensorDescriptor> parameters) {
    public ModelLayout {
        Objects.requireNonNull(settings, "settings");
        parameters = Map.copyOf(parameters);
    }

    public ParameterFactory parameterFactory() {
        return parameters.isEmpty() ? ParameterFactory.float32() : new ParameterFactory(parameters);
    }

    public void requireMatches(List<Parameter> actual) throws IOException {
        if (parameters.isEmpty()) return;
        var names = actual.stream().map(Parameter::name).collect(Collectors.toSet());
        if (actual.size() != parameters.size() || !names.equals(parameters.keySet())) {
            throw new IOException("Checkpoint parameter names do not match the model");
        }
        for (Parameter parameter : actual) { parameters.get(parameter.name()).requireMatches(parameter.value); }
    }
}
