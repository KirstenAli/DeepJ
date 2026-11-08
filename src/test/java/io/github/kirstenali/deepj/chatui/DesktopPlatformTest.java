package io.github.kirstenali.deepj.chatui;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DesktopPlatformTest {
    @Test
    void onlyAppleSiliconUsesTheBundledMetalLibrary() {
        assertTrue(DesktopPlatform.supportsMetal("Mac OS X", "aarch64"));
        assertFalse(DesktopPlatform.supportsMetal("Windows 11", "amd64"));
        assertFalse(DesktopPlatform.supportsMetal("Linux", "aarch64"));
        assertFalse(DesktopPlatform.supportsMetal("Mac OS X", "x86_64"));
        assertFalse(DesktopPlatform.supportsMetal(null, null));
    }
}
