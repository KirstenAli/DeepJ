package io.github.kirstenali.deepj.chatui;

import netscape.javascript.JSObject;

final class DesktopBridges {
    private DesktopBridges() {}

    static void connect(JSObject window, ChatWindow chat, ImageWorkspace images,
                        VideoWorkspace videos, AgentWorkspace agent) {
        window.setMember("chatBridge", chat);
        window.setMember("imageBridge", images);
        window.setMember("videoBridge", videos);
        if (agent != null) window.setMember("agentBridge", agent);
        window.call("bridgeReady");
    }
}
