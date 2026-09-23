package io.github.kirstenali.deepj.chatui;

public class ChatApp extends BaseChatApp {

    @Override
    protected ChatService createChatService() {
        return new DeepJOriginChatService();
    }

    public static void main(String[] args) {
        launch();
    }
}
