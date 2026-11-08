package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

final class DesktopPlatform {
    private DesktopPlatform() {}

    static boolean metalAvailable() {
        return supportsMetal(System.getProperty("os.name"), System.getProperty("os.arch"))
                && MetalBackend.isAvailable();
    }

    static boolean supportsMetal(String operatingSystem, String architecture) {
        return "Mac OS X".equals(operatingSystem) && "aarch64".equals(architecture);
    }
}
