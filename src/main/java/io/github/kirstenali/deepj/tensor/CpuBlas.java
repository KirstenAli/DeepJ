package io.github.kirstenali.deepj.tensor;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/** Optional CPU library, loaded independently of the Metal backend. */
final class CpuBlas {
    static final boolean AVAILABLE = load();

    private CpuBlas() {}

    private static boolean load() {
        if (Boolean.getBoolean("deepj.cpu.javaOnly") || !platform()) return false;
        try (InputStream in = CpuBlas.class.getResourceAsStream("/native/macos/libdeepj_cpu_jni.dylib")) {
            if (in == null) return false;
            Path file = Files.createTempFile("deepj_cpu_", ".dylib");
            file.toFile().deleteOnExit();
            Files.copy(in, file, StandardCopyOption.REPLACE_EXISTING);
            System.load(file.toAbsolutePath().toString());
            return true;
        } catch (java.io.IOException | UnsatisfiedLinkError | SecurityException unavailable) { return false; }
    }

    private static boolean platform() {
        if (!System.getProperty("os.name", "").equals("Mac OS X")) return false;
        if (!System.getProperty("os.arch", "").equals("aarch64")) return false;
        try { return Integer.parseInt(System.getProperty("os.version", "0").split("\\.")[0]) >= 15; }
        catch (NumberFormatException invalid) { return false; }
    }

    static native void product(float[] left, float[] right, float[] output, int[] shape);
    static native void compactProduct(Object left, Object right, Object output, int[] shape);
}
