package io.github.kirstenali.deepj.onnx;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Registry entries include operator-version ranges and accepted attributes. */
public final class OnnxRegistry {
    record Rule(long first, long last, Set<String> attributes, OnnxOperator operator) {}

    private final Map<String, Rule> rules = new LinkedHashMap<>();

    public static OnnxRegistry standard() {
        var registry = new OnnxRegistry();
        ArithmeticOperators.register(registry);
        ShapeOperators.register(registry);
        ReductionOperators.register(registry);
        NeuralOperators.register(registry);
        SpatialOperators.register(registry);
        RecurrentOperators.register(registry);
        ControlOperators.register(registry);
        return registry;
    }

    public OnnxRegistry register(
            String domain,
            String type,
            long first,
            long last,
            Set<String> attributes,
            OnnxOperator operator) {
        if (first < 1 || last < first)
            throw new IllegalArgumentException("Invalid ONNX operator-version range");
        var key = domain(domain) + ":" + type;
        if (rules.putIfAbsent(
                        key,
                        new Rule(
                                first,
                                last,
                                Set.copyOf(attributes),
                                Objects.requireNonNull(operator)))
                != null)
            throw new IllegalArgumentException("ONNX operator already registered: " + key);
        return this;
    }

    void add(String type, long first, String attributes, OnnxOperator operator) {
        register(
                "",
                type,
                first,
                28,
                attributes.isBlank() ? Set.of() : Set.of(attributes.split(" ")),
                operator);
    }

    Rule find(String domain, String type) {
        return rules.get(domain(domain) + ":" + type);
    }

    static String domain(String domain) {
        return domain.equals("ai.onnx") ? "" : domain;
    }

    public Set<String> operators() {
        return Set.copyOf(rules.keySet());
    }
}
