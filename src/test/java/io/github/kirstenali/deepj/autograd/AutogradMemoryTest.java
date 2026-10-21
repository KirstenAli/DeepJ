package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AutogradMemoryTest {
    @TempDir
    Path temporary;

    @ParameterizedTest
    @ValueSource(strings = {"shift", "softmax"})
    void shapeOnlyGraphEdgesDoNotRetainFullActivationArrays(String mode) throws Exception {
        Path log = temporary.resolve("memory-probe.txt");
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        String classpath = location(AutogradMemoryProbe.class) + File.pathSeparator + location(Tensor.class);
        Process process = new ProcessBuilder(java, "-Xmx96m", "-cp", classpath,
                AutogradMemoryProbe.class.getName(), mode)
                .redirectErrorStream(true).redirectOutput(log.toFile()).start();
        boolean completed = process.waitFor(30, TimeUnit.SECONDS);
        if (!completed) process.destroyForcibly();
        assertTrue(completed, "Memory probe timed out");
        assertEquals(0, process.exitValue(), Files.readString(log));
    }

    private static String location(Class<?> type) throws Exception {
        return Path.of(type.getProtectionDomain().getCodeSource().getLocation().toURI()).toString();
    }
}
