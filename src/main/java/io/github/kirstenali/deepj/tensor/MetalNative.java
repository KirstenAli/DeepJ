package io.github.kirstenali.deepj.tensor;

final class MetalNative {

    static {
        System.loadLibrary("deepj_metal_jni");
    }

    private MetalNative() {}

    static native void matmulF32(
            float[] a,
            float[] b,
            float[] out,
            int m,
            int n,
            int k
    );
}