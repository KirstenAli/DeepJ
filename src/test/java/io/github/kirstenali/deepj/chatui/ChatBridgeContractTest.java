package io.github.kirstenali.deepj.chatui;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ChatBridgeContractTest {

    private static final Class<?>[] SETTINGS = {
            String.class, int.class, double.class, int.class, long.class, double.class
    };

    @Test
    void generationBridgeHasOneUnambiguousSignature() {
        var methods = Arrays.stream(ChatWindow.class.getMethods())
                .filter(method -> method.getName().equals("sendWithSettings")).toList();
        assertEquals(1, methods.size());
        assertArrayEquals(SETTINGS, methods.get(0).getParameterTypes());
        assertEquals(void.class, methods.get(0).getReturnType());
    }

    @Test
    void existingJavaSendSignaturesRemainAvailable() throws Exception {
        assertEquals(void.class, ChatWindow.class.getMethod("send", Arrays.copyOf(SETTINGS, 5)).getReturnType());
        assertEquals(void.class, ChatWindow.class.getMethod("send", SETTINGS).getReturnType());
    }
}
