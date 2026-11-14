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

    static native void normalize(float[] input, float[] gamma, float[] beta, float[] output,
                                 float[] normalized, float[] denominator, int[] shape, float epsilon);
    static native void normalizationBackward(float[] gradient, float[] normalized, float[] denominator,
                                              float[] gamma, float[] output, int[] shape);
    static native void compactNormalize(Object input, Object gamma, Object beta, Object output,
                                         float[] normalized, float[] denominator, int[] shape, float epsilon);
    static native void compactNormalizationBackward(Object gradient, float[] normalized, float[] denominator,
                                                      Object gamma, float[] output, int[] shape);
    static native boolean causalSoftmax(float[] input, float[] output, int rows, int width,
                                         float scale, int workers, int[] fallback);
    static native void q8Project(short[] packed, float[] input, float[] output, int rows, int width,
                                 int batch, int workers);
    static native void product(float[] left, float[] right, float[] output, int[] shape);
    static native void compactProduct(Object left, Object right, Object output, int[] shape);
    static native boolean compactSoftmax(short[] input, short[] output, int[] shape, float scale, int[] fallback);
    static native boolean compactSwiGluBackward(Object gradient, Object gate, Object up,
                                                float[] gateGradient, float[] upGradient, int[] shape);
    static native boolean compactPointwise(Object left, Object right, Object output, int[] shape);
    static native boolean swiGlu(float[] gate, int go, float[] up, int uo, float[] output, int oo, int count);
    static native boolean matrixAdd(float[] left, float[] right, float[] output, int[] shape);
    static native float sumSquares(float[] values, float initial);
    static native boolean adamW(float[] weight, int wo, float[] gradient, int go, float[] first, int mo,
                                 float[] second, int vo, int count, float learningRate, float beta1, float beta2,
                                 float epsilon, float decay, float bc1, float bc2);
}
