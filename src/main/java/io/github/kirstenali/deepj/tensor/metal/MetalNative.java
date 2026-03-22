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
        // Extract from classpath resources then System.load(absolutePath).
        // A native library must be a real file on disk; it can't be loaded directly from a jar.
        final String resourcePath = "/native/macos/libdeepj_metal_jni.dylib";

        try (InputStream in = MetalNative.class.getResourceAsStream(resourcePath)) {
            if (in == null) return false;

            // The prefix is only used for the extracted temp filename (OS adds a unique suffix).
            Path tmp = Files.createTempFile("deepj_metal_jni_", ".dylib");
            // Ensure the extracted binary isn't removed before load completes.
            tmp.toFile().deleteOnExit();
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            System.load(tmp.toAbsolutePath().toString());
            return true;
        } catch (Throwable ignored) {
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