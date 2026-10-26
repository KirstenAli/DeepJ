package io.github.kirstenali.deepj.tensor.metal;

/** Keep old FLOAT32-index kernels only where their index transport is lossless. */
final class MetalIndexMetadata {
    private MetalIndexMetadata() {}

    static boolean fitsFloat(int[] indices) {
        for (int index : indices) {
            if ((double) (float) index != index) return false;
        }
        return true;
    }
}
