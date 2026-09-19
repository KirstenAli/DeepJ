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

    static native void nativeAllocBuffers(int[] ids, int[] sizes, int count);

    static native void nativeUploadBuffer(int bufId, float[] data);

    static native void nativeDownloadBuffer(int bufId, float[] out);

    static native void nativeReleaseBuffers(int[] ids, int count);

    static native void nativeFlushOps(int[] cmdStream, int cmdStreamLength);

}
