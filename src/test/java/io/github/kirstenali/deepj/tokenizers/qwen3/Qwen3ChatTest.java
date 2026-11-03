package io.github.kirstenali.deepj.tokenizers.qwen3;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3ChatTest {
    @Test void textConversationFormattingMatchesOfficialJinjaIncludingPriorAndCurrentReasoning()
            throws IOException, URISyntaxException {
        var file = Path.of(getClass().getResource("/qwen3/chat-golden.json").toURI());
        for (Object item : Qwen3Json.array(Qwen3Json.read(file).get("cases"))) {
            var expected = Qwen3Json.object(item);
            List<Qwen3Chat.Message> messages = Qwen3Json.array(expected.get("messages")).stream()
                    .map(Qwen3Json::object).map(Qwen3ChatTest::message).toList();
            assertEquals(expected.get("expected"), Qwen3Chat.format(messages,
                    Boolean.TRUE.equals(expected.get("generation")), Boolean.TRUE.equals(expected.get("thinking"))));
        }
    }

    private static Qwen3Chat.Message message(java.util.Map<String, Object> values) {
        String reasoning = values.containsKey("reasoning_content")
                ? Qwen3Json.string(values.get("reasoning_content")) : null;
        return new Qwen3Chat.Message(Qwen3Chat.Role.valueOf(Qwen3Json.string(values.get("role")).toUpperCase()),
                Qwen3Json.string(values.get("content")), reasoning);
    }
}
