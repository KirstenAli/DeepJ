package io.github.kirstenali.deepj.onnx;

import java.util.List;

/** Explicit extension point for application-provided ONNX operators. */
@FunctionalInterface
public interface OnnxOperator {
    List<OnnxValue> apply(OnnxContext context);
}
