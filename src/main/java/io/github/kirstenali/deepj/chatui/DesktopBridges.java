package io.github.kirstenali.deepj.chatui;

import netscape.javascript.JSObject;

final class DesktopBridges {
    private DesktopBridges() {}

    static void connect(JSObject window, ChatWindow chat, ImageWorkspace images,
                        VideoWorkspace videos, AgentWorkspace agent, DesktopExecution execution) {
        window.setMember("chatBridge", chat);
        window.setMember("imageBridge", images);
        window.setMember("videoBridge", videos);
        window.setMember("executionBridge", execution);
        if (agent != null) window.setMember("agentBridge", agent);
        window.call("bridgeReady");
        window.call("executionReady");
    }
}
