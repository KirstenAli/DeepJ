package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Arbitrary-axis APIs leave the established full FLOAT32 matrix reduction kernels available. */
abstract class MetalAxisOps extends MetalCoreOps {
    @Override public Tensor sumAxes(Tensor input, int[] axes, boolean keep) {
        return graph.layoutReduction(input, axes, keep, 0, 0);
    }

    @Override public Tensor meanAxes(Tensor input, int[] axes, boolean keep) {
        return graph.layoutReduction(input, axes, keep, 1, 0);
    }

    @Override public Tensor varianceAxes(Tensor input, int[] axes, boolean keep, int correction) {
        return graph.layoutReduction(input, axes, keep, 2, correction);
    }

    @Override public Tensor maximumAxes(Tensor input, int[] axes, boolean keep) {
        return graph.layoutReduction(input, axes, keep, 3, 0);
    }

    @Override public Tensor maximumAxesBackward(Tensor input, Tensor maximum, Tensor gradient,
                                               int[] axes, boolean keep) {
        return graph.layoutMaximumGradient(input, maximum, gradient, axes, keep);
    }

    @Override public Tensor softmaxAxis(Tensor input, int axis) { return graph.layoutSoftmax(input, axis); }

    @Override public Tensor sliceGradient(Tensor gradient, int[] shape, int axis, int start, int step) {
        return graph.layoutSliceGradient(gradient, shape, axis, start, step);
    }

    @Override public Tensor indexSelect(Tensor input, int axis, int[] indices) {
        return graph.layoutIndexSelect(input, axis, indices);
    }

    @Override public void indexAdd(Tensor target, int axis, int[] indices, Tensor updates) {
        graph.layoutIndexAdd(target, axis, indices, updates);
    }

    @Override public Tensor indexSelectGradient(Tensor gradient, int[] shape, int axis, int[] indices) {
        return graph.layoutIndexGradient(gradient, shape, axis, indices);
    }

    final Tensor axisSoftmaxBackward(Tensor gradient, Tensor probabilities, int axis) {
        Tensor.requireSameShape(gradient, probabilities, "Softmax gradient");
        Tensor product = multiply(gradient, probabilities);
        return multiply(probabilities, subtract(gradient, sumAxes(product, new int[]{axis}, true)));
    }
}
