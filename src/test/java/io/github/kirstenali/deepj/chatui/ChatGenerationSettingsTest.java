package io.github.kirstenali.deepj.chatui;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ChatGenerationSettingsTest {

    @Test
    void penaltyAlternativePreservesOtherSettings() {
        ChatGenerationSettings original = new ChatGenerationSettings(80, 0.8, 40, 1234, 1.0);
        assertEquals(new ChatGenerationSettings(80, 0.8, 40, 1234, 1.1), original.withPenalty());
    }

    @Test
    void seedAlternativePreservesOtherSettings() {
        ChatGenerationSettings original = new ChatGenerationSettings(80, 0.8, 40, 1234, 1.0);
        assertEquals(new ChatGenerationSettings(80, 0.8, 40, 1235, 1.0), original.withNewSeed());
    }
}
