package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorDescriptor;

import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/** Explicit, immutable parameter storage policy. Names do not depend on allocation order. */
public final class ParameterFactory {
    private final String prefix;
    private final DType defaultType;
    private final Map<String, TensorDescriptor> descriptors;
    private final boolean randomInitialization;

    public ParameterFactory(DType dtype) {
        this("", Objects.requireNonNull(dtype, "dtype"), Map.of(), true);
    }

    public ParameterFactory(Map<String, TensorDescriptor> descriptors) {
        this("", null, Map.copyOf(descriptors), true);
    }

    private ParameterFactory(String prefix, DType defaultType, Map<String, TensorDescriptor> descriptors,
                              boolean randomInitialization) {
        this.prefix = prefix;
        this.defaultType = defaultType;
        this.descriptors = descriptors;
        this.randomInitialization = randomInitialization;
    }

    public static ParameterFactory float32() { return new ParameterFactory(DType.FLOAT32); }

    /** Zero-filled random-weight slots for a complete checkpoint replacement, without device initialization. */
    public ParameterFactory forLoading() {
        return new ParameterFactory(prefix, defaultType, descriptors, false);
    }

    public ParameterFactory scope(String name) {
        return new ParameterFactory(path(name), defaultType, descriptors, randomInitialization);
    }

    public Parameter random(String name, int rows, int columns, Random random) {
        DType dtype = dtype(name, rows, columns);
        if (!randomInitialization) return new Parameter(path(name), new Tensor(dtype, rows, columns));
        Tensor value = dtype == DType.FLOAT32 ? Tensor.random(rows, columns, random)
                : Tensor.random(dtype, random, rows, columns);
        return new Parameter(path(name), value);
    }

    public Parameter zeros(String name, int rows, int columns) {
        return new Parameter(path(name), Tensor.zeros(dtype(name, rows, columns), rows, columns));
    }

    public Parameter ones(String name, int rows, int columns) {
        return new Parameter(path(name), Tensor.ones(dtype(name, rows, columns), rows, columns));
    }

    private DType dtype(String name, int rows, int columns) {
        if (rows < 1 || columns < 1) throw new IllegalArgumentException("Parameter dimensions must be positive");
        if (defaultType != null) return defaultType;
        TensorDescriptor descriptor = descriptors.get(path(name));
        if (descriptor == null) throw new IllegalArgumentException("Missing parameter: " + path(name));
        if (!Arrays.equals(descriptor.shape(), new int[]{rows, columns})) {
            throw new IllegalArgumentException("Parameter shape mismatch: " + path(name));
        }
        return descriptor.dtype();
    }

    private String path(String name) {
        if (name == null || !name.matches("[A-Za-z0-9_]+(?:\\.[A-Za-z0-9_]+)*")) {
            throw new IllegalArgumentException("Parameter names must contain dot-separated identifiers");
        }
        return prefix.isEmpty() ? name : prefix + "." + name;
    }
}
