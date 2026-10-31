package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

final class MetalNative {

    static final boolean AVAILABLE;
    static final boolean TILED_ATTENTION_AVAILABLE;

    static {
        AVAILABLE = loadNative();
        TILED_ATTENTION_AVAILABLE = AVAILABLE && nativeSupportsTiledAttention();
    }

    private MetalNative() {}

    private static boolean loadNative() {

        final String resourcePath = "/native/macos/libdeepj_metal_jni.dylib";

        try (InputStream in = MetalNative.class.getResourceAsStream(resourcePath)) {
            if (in == null) return false;

            Path tmp = Files.createTempFile("deepj_metal_jni_", ".dylib");

            tmp.toFile().deleteOnExit();
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            System.load(tmp.toAbsolutePath().toString());
            return nativeIsAvailable();
        } catch (Throwable t) {
            System.err.println("[DeepJ/Metal] Failed to load native library from: " + resourcePath);
            t.printStackTrace(System.err);
            return false;
        }
    }

    private static native boolean nativeIsAvailable();

    private static native boolean nativeSupportsTiledAttention();

    static boolean supportsAttention(DType dtype) {
        if (dtype == DType.FLOAT32) return TILED_ATTENTION_AVAILABLE;
        return dtype == DType.FLOAT16 ? CompactAttention.HALF : CompactAttention.BFLOAT;
    }

    private static final class CompactAttention {
        private static final boolean HALF = AVAILABLE && nativeSupportsTypedAttention(2);
        private static final boolean BFLOAT = AVAILABLE && nativeSupportsTypedAttention(3);

        private CompactAttention() {}
    }

    private static native boolean nativeSupportsTypedAttention(int dtype);

    static native boolean nativeSupportsCompactMatmul(int dtype);

    static native void nativeAllocBuffers(int[] ids, int[] sizes, int count);
    static native void nativeAllocTypedBuffers(int[] ids, int[] elements, int[] dtypes, int count);

    static native void nativeUploadBuffer(int bufId, float[] data);
    static native void nativeUploadBuffer16(int bufId, short[] data);

    static native void nativeZeroBuffer(int bufId, int floatCount);
    static native void nativeZeroTypedBuffer(int bufId, int elements, int dtype);

    static native void nativeDownloadBuffer(int bufId, float[] out);
    static native void nativeDownloadBuffer16(int bufId, short[] out);
    static native long nativeBufferByteLength(int bufId);

    static native void nativeReleaseBuffers(int[] ids, int count);

    static native void nativeFlushOps(int[] cmdStream, int cmdStreamLength);
    static native void nativeFlushLayoutOps(int[] commands, int length, long[] layouts);

}
