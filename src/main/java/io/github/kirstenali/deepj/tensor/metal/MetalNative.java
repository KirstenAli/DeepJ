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

    static native void matmulF32(
            float[] a,
            float[] b,
            float[] out,
            int m,
            int n,
            int k
    );
}