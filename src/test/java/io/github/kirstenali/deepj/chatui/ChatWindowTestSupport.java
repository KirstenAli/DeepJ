package io.github.kirstenali.deepj.chatui;

import javafx.scene.web.WebView;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.function.Consumer;

final class ChatWindowTestSupport {

    private ChatWindowTestSupport() {}

    static Object bridgeState(WebView view) {
        return view.getEngine().executeScript("JSON.stringify({status:document.getElementById('status').textContent,"
                + "busy:busy,loaded:modelLoaded,prompt:promptBox.value,html:messages.innerHTML})");
    }

    static void startAlternativeConversation(WebView view) {
        view.getEngine().executeScript("setModel('stub.dj'); chatBridge.send('Hello', 80, 0.8, 40, 1234, 1)");
        System.out.println("Alternative bridge state: " + bridgeState(view));
    }

    record Invocation(String prompt, int maxTokens, float temperature, int topK,
                      long seed, float penalty) {}

    static final class TestService implements ChatService {
        final CountDownLatch generated = new CountDownLatch(1);
        volatile float repetitionPenalty;

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
