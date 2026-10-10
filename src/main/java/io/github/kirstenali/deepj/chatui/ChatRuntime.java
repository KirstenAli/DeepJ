package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.tensor.Tensor;
import javafx.concurrent.Task;

final class ChatRuntime {

    private ChatRuntime() {
    }

    static String backendName() {
        return Tensor.backend().getClass().getSimpleName();
    }

    static String errorMessage(Throwable error) {
        if (missingMessage(error)) return "Unknown error";
        return error.getMessage();
    }

    static void start(Task<?> task, String name) {
        Thread thread = new Thread(task, name);
        thread.setDaemon(true);
        thread.start();
    }

    private static boolean missingMessage(Throwable error) {
        return error == null || error.getMessage() == null || error.getMessage().isBlank();
    }
}
