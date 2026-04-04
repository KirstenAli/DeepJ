package io.github.kirstenali.deepj.tensor.metal;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

final class MetalNative {

    static final boolean AVAILABLE;

    static {
        AVAILABLE = loadNative();
    }

    private MetalNative() {}

    private static boolean loadNative() {
        // A native library must be a real file on disk; it can't be loaded directly from a jar.
        final String resourcePath = "/native/macos/libdeepj_metal_jni.dylib";

        try (InputStream in = MetalNative.class.getResourceAsStream(resourcePath)) {
            if (in == null) return false;

            Path tmp = Files.createTempFile("deepj_metal_jni_", ".dylib");

            tmp.toFile().deleteOnExit();
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            System.load(tmp.toAbsolutePath().toString());
            return true;
        } catch (Throwable t) {
            System.err.println("[DeepJ/Metal] Failed to load native library from: " + resourcePath);
            t.printStackTrace(System.err);
            return false;
        }
    }

    // ── matmul ─────────────────────────────────────────────────────────
    static native void matmulF32(float[] a, float[] b, float[] out, int m, int n, int k);

    // ── element-wise binary ────────────────────────────────────────────
    static native void addF32(float[] a, float[] b, float[] out, int n);
    static native void subtractF32(float[] a, float[] b, float[] out, int n);
    static native void multiplyF32(float[] a, float[] b, float[] out, int n);
    static native void divideF32(float[] a, float[] b, float[] out, int n);

    // ── scalar ops ─────────────────────────────────────────────────────
    static native void multiplyScalarF32(float[] a, float[] out, float scalar, int n);

    // ── unary math ─────────────────────────────────────────────────────
    static native void sqrtF32(float[] a, float[] out, int n);
    static native void negF32(float[] a, float[] out, int n);
    static native void expF32(float[] a, float[] out, int n);
    static native void logF32(float[] a, float[] out, int n);

    // ── activations ────────────────────────────────────────────────────
    static native void tanhF32(float[] a, float[] out, int n);
    static native void sigmoidF32(float[] a, float[] out, int n);
    static native void reluF32(float[] a, float[] out, int n);
    static native void reluBackwardF32(float[] input, float[] gradOutput, float[] out, int n);
    static native void geluF32(float[] a, float[] out, int n);
    static native void geluBackwardF32(float[] input, float[] gradOutput, float[] out, int n);

    // ── row-wise compound ──────────────────────────────────────────────
    static native void softmaxRowsF32(float[] a, float[] out, int rows, int cols);

    // ═══════════════════════════════════════════════════════════════════
    //  Lazy graph execution: persistent GPU buffers + batch op flush
    // ═══════════════════════════════════════════════════════════════════

    /** Allocate multiple GPU buffers in one call. ids[i] → buffer of sizes[i] floats. */
    static native void nativeAllocBuffers(int[] ids, int[] sizes, int count);

    /** Upload CPU float data into a GPU buffer. */
    static native void nativeUploadBuffer(int bufId, float[] data);

    /** Download GPU buffer contents to CPU float array. */
    static native void nativeDownloadBuffer(int bufId, float[] out);

    /** Release multiple GPU buffers. */
    static native void nativeReleaseBuffers(int[] ids, int count);

    /**
     * Execute a batch of ops encoded as a flat int[] command stream, all in one
     * MTLCommandBuffer. Op format: [opCode, args...] where arg count depends on opCode.
     * Buffer IDs in the stream reference previously allocated GPU buffers.
     */
    static native void nativeFlushOps(int[] cmdStream, int cmdStreamLength);

}