package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;

import javafx.scene.web.WebView;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.function.Consumer;

final class ChatWindowTestSupport {

    private ChatWindowTestSupport() {}

    static void startAlternativeConversation(WebView view) {
        view.getEngine().executeScript("setModel('stub.dj');"
                + "chatBridge.sendWithSettings('Hello', 80, 0.8, 40, 1234, 1)");
    }

    record Invocation(String prompt, int maxTokens, float temperature, int topK,
                      long seed, float penalty) {}

    static final class TestService implements ChatService {
        final CountDownLatch generated = new CountDownLatch(1);
        volatile float repetitionPenalty;
        volatile ContextWindowMode contextWindow;

        @Override
        public String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                               float penalty, ContextWindowMode mode, Consumer<String> listener) {
            contextWindow = mode;
            return generate(prompt, maxTokens, temperature, topK, seed, penalty, listener);
        }

        @Override public void loadModel(Path modelPath) {}
        @Override public boolean isModelLoaded() { return true; }
        @Override public String getLoadedModelName() { return "stub.dj"; }

        @Override
        public String generate(String prompt, int maxTokens, float temperature, int topK,
                               long seed, Consumer<String> listener) {
            listener.accept("A short");
            generated.countDown();
            return "A short answer";
        }

        @Override
        public String generate(String prompt, int maxTokens, float temperature, int topK,
                               long seed, float penalty, Consumer<String> listener) {
            repetitionPenalty = penalty;
            return generate(prompt, maxTokens, temperature, topK, seed, listener);
        }
    }

    static final class AlternativeService implements ChatService {
        final CountDownLatch first = new CountDownLatch(1);
        final CountDownLatch extra = new CountDownLatch(2);
        final List<Invocation> calls = new CopyOnWriteArrayList<>();
        final List<ContextWindowMode> modes = new CopyOnWriteArrayList<>();

        @Override
        public String generate(String prompt, int maxTokens, float temperature, int topK, long seed,
                               float penalty, ContextWindowMode mode, Consumer<String> listener) {
            modes.add(mode);
            return generate(prompt, maxTokens, temperature, topK, seed, penalty, listener);
        }

        @Override public void loadModel(Path modelPath) {}
        @Override public boolean isModelLoaded() { return true; }
        @Override public String getLoadedModelName() { return "stub.dj"; }

        @Override
        public String generate(String prompt, int maxTokens, float temperature, int topK,
                               long seed, Consumer<String> listener) {
            return generate(prompt, maxTokens, temperature, topK, seed, 1.0f, listener);
        }

        @Override
        public String generate(String prompt, int maxTokens, float temperature, int topK,
                               long seed, float penalty, Consumer<String> listener) {
            calls.add(new Invocation(prompt, maxTokens, temperature, topK, seed, penalty));
            String answer = "response-" + calls.size();
            listener.accept(answer);
            if (calls.size() == 1) first.countDown();
            else extra.countDown();
            return answer;
        }
    }

    static final class BlockingService implements ChatService {
        final CountDownLatch started = new CountDownLatch(1);
        final CountDownLatch stopped = new CountDownLatch(1);
        final CountDownLatch finished = new CountDownLatch(1);

        @Override public void loadModel(Path modelPath) {}
        @Override public boolean isModelLoaded() { return true; }
        @Override public String getLoadedModelName() { return "stub.dj"; }

        @Override
        public String generate(String prompt, int maxTokens, float temperature, int topK,
                               long seed, Consumer<String> listener) {
            if (prompt.equals("new")) {
                finished.countDown();
                return "new response";
            }
            started.countDown();
            try {
                Thread.sleep(30000);
            } catch (InterruptedException ignored) {
                stopped.countDown();
            }
            return "stale response";
        }
    }
}
