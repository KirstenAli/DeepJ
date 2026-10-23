package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AutogradMemoryTest {
    @TempDir
    Path temporary;

    @ParameterizedTest
    @ValueSource(strings = {"shift", "softmax"})
    void shapeOnlyGraphEdgesDoNotRetainFullActivationArrays(String mode) throws Exception {
        assertProbe(AutogradMemoryProbe.class, mode, "-Xmx96m");
    }

    @Test
    void leafGradientTemporariesAreReleasedDuringBackward() throws Exception {
        assertProbe(LeafGradientMemoryProbe.class, "leaf", "-Xmx128m", "-XX:+UseSerialGC");
    }

    private void assertProbe(Class<?> probe, String mode, String... options) throws Exception {
        Path log = temporary.resolve(mode + "-memory-probe.txt");
        Process process = startProbe(probe, mode, log, options);
        boolean completed = process.waitFor(30, TimeUnit.SECONDS);
        if (!completed) process.destroyForcibly();
        assertTrue(completed, "Memory probe timed out");
        assertEquals(0, process.exitValue(), Files.readString(log));
    }

    private static Process startProbe(Class<?> probe, String mode, Path log, String[] options) throws Exception {
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        String classpath = location(probe) + File.pathSeparator + location(Tensor.class);
        var command = new ArrayList<>(List.of(java));
        command.addAll(List.of(options));
        command.addAll(List.of("-cp", classpath, probe.getName(), mode));
        return new ProcessBuilder(command).redirectErrorStream(true).redirectOutput(log.toFile()).start();
    }

    private static String location(Class<?> type) throws Exception {
        return Path.of(type.getProtectionDomain().getCodeSource().getLocation().toURI()).toString();
    }
}
