package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.MetalUnavailableClassLoader;
import org.junit.jupiter.api.Test;

import java.lang.reflect.InvocationTargetException;

import static org.junit.jupiter.api.Assertions.*;

class ChatCpuFallbackTest {

    @Test
    void automaticSelectionUsesCpuWhenTheMetalLibraryIsMissing() throws Exception {
        try (var isolated = new MetalUnavailableClassLoader()) {
            assertNull(isolated.getResourceAsStream("native/macos/libdeepj_metal_jni.dylib"));
            assertEquals(false, isolated.method("tensor.metal.MetalCoreOps", "isAvailable").invoke(null));
            isolated.method("chatui.ChatBackend", "configure").invoke(null);
            Object backend = isolated.method("tensor.Tensor", "backend").invoke(null);
            assertSame(isolated, backend.getClass().getClassLoader());
            assertEquals("io.github.kirstenali.deepj.tensor.cpu.CpuBackend", backend.getClass().getName());
            assertEquals("Ready on CPU", isolated.method("chatui.ChatBackend", "readyStatus").invoke(null));
        }
    }

    @Test
    void explicitlyRequestingMetalDoesNotSilentlySubstituteCpu() throws Exception {
        try (var isolated = new MetalUnavailableClassLoader()) {
            var constructor = isolated.loadClass("io.github.kirstenali.deepj.tensor.metal.MetalBackend")
                    .getConstructor();
            var failure = assertThrows(InvocationTargetException.class, constructor::newInstance);
            assertInstanceOf(IllegalStateException.class, failure.getCause());
            assertEquals("Metal is not available on this system", failure.getCause().getMessage());
        }
    }
}
