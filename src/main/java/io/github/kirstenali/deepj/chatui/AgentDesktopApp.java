package io.github.kirstenali.deepj.chatui;

/** Compatibility entry point for the unified desktop chat application. */
public final class AgentDesktopApp extends ChatApp {

    public static void main(String[] args) { launch(args); }

    @Override
    protected String getAppTitle() { return "DeepJ Agent"; }
}
