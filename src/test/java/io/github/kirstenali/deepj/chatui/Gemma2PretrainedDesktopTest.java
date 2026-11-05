package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

/** Opt-in full pretrained chat, exercised through the production loader, controller and WebView. */
@EnabledIfSystemProperty(named = "deepj.gemma2Checkpoint", matches = ".+")
class Gemma2PretrainedDesktopTest extends ChatDesktopTestSupport {
    @Test void officialGemmaWeightsStreamAReplyThroughTheNormalDesktopChat() throws Exception {
        Tensor.setBackend(new MetalBackend());
        var service = new DeepJChatService();
        try {
            service.loadModel(Path.of(System.getProperty("deepj.gemma2Checkpoint"), "config.json"));
            var fixture = open(service);
            try {
                verifyWithoutClosing(fixture, view -> script(view, "setModel('Gemma 2 2B-IT');"
                        + "setRollingCacheAvailable(false);"
                        + "setTheme('dark'); setStatus('Ready on Metal');"
                        + "chatBridge.sendWithWindowSettings('Explain what a tensor is in one sentence.',"
                        + "80,1,1,2,1,false)"));
                waitForReply(fixture);
                verifyWithoutClosing(fixture, this::checkReply);
                ImageDesktopSnapshots.capture(fixture, "gemma-chat");
            } finally { verify(fixture, view -> {}); }
        } finally { Tensor.backend().releaseResources(); }
    }

    private static void waitForReply(Fixture fixture) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(3);
        var finished = new java.util.concurrent.atomic.AtomicBoolean();
        while (!finished.get() && System.nanoTime() < deadline) {
            verifyWithoutClosing(fixture, view -> finished.set(Boolean.TRUE.equals(script(view,
                    "document.querySelector('.assistant') !== null && !element('prompt').disabled"))));
            Thread.sleep(100);
        }
        assertTrue(finished.get(), "Gemma desktop generation did not complete");
    }

    private void checkReply(javafx.scene.web.WebView view) {
        String reply = (String) script(view, "document.querySelector('.assistant .message-content').textContent");
        assertFalse(reply.contains("Generation failed"), reply);
        assertTrue(reply.toLowerCase(java.util.Locale.ROOT).contains("tensor"), reply);
        assertTrue(reply.length() > 30, reply);
        try {
            Files.createDirectories(ImageDesktopSnapshots.output());
            Files.writeString(ImageDesktopSnapshots.output().resolve("gemma-reply.txt"), reply);
        }
        catch (java.io.IOException failure) { throw new AssertionError(failure); }
    }
}
