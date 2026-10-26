package io.github.kirstenali.deepj.tensor;

/** Shared CPU/GPU preflight for the exact logical region addressed by a slice adjoint. */
final class SliceGradientLayout {
    private SliceGradientLayout() {}

    static TensorLayout of(Tensor gradient, int[] dimensions, int axis, int start, int step) {
        TensorStorage.requireSource(gradient);
        TensorShape shape = new TensorShape(dimensions);
        axis = shape.axis(axis);
        long end = gradient.size(axis) == 0 ? start : start + (gradient.size(axis) - 1L) * step + 1;
        TensorLayout region = LayoutTransforms.slice(TensorLayout.dense(shape),
                axis, start, Math.toIntExact(end), step);
        if (!region.shape().equals(gradient.layout().shape())) {
            throw new IllegalArgumentException("Slice gradient must match the selected region");
        }
        return region;
    }
}
