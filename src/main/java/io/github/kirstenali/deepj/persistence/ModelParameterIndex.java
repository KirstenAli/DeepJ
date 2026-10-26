package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.Parameter;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Named model files resolve values by semantic identity; old files retain positional framing. */
final class ModelParameterIndex {
    private final List<Parameter> parameters;
    private final Map<String, Parameter> named;

    ModelParameterIndex(List<Parameter> parameters, ModelFormat format) throws IOException {
        this.parameters = parameters;
        named = format.named() ? index(parameters) : null;
    }

    private static Map<String, Parameter> index(List<Parameter> parameters) throws IOException {
        Map<String, Parameter> named = new HashMap<>();
        for (Parameter parameter : parameters) {
            if (parameter.name().isEmpty() || named.putIfAbsent(parameter.name(), parameter) != null) {
                throw new IOException("Named checkpoints require unique destination parameter names");
            }
        }
        return named;
    }

    Parameter match(int position, ModelTensorIO.Entry entry) throws IOException {
        Parameter parameter = named == null ? parameters.get(position) : named.get(entry.name());
        if (parameter == null) throw new IOException("Unknown checkpoint parameter: " + entry.name());
        entry.requireMatches(parameter);
        return parameter;
    }
}
