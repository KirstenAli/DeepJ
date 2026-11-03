package io.github.kirstenali.deepj.tokenizers.qwen3;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Qwen3TokenizerFixture {
    private Qwen3TokenizerFixture() {}

    public static void write(Path directory) throws IOException {
        Files.writeString(directory.resolve("tokenizer.json"), Qwen3Json.stringify(recipe()));
    }

    static Map<String, Object> recipe() {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("normalizer", Map.of("type", "NFC"));
        root.put("decoder", Map.of("type", "ByteLevel"));
        root.put("pre_tokenizer", Map.of("type", "Sequence", "pretokenizers", List.of(
                Map.of("type", "Split", "pattern", Map.of("Regex", "(?s).+"), "behavior", "Isolated"),
                Map.of("type", "ByteLevel", "add_prefix_space", false, "use_regex", false))));
        Map<String, Integer> vocabulary = new LinkedHashMap<>();
        var alphabet = new ByteAlphabet();
        for (int index = 0; index < 256; index++) { vocabulary.put(alphabet.symbol(index), index); }
        vocabulary.put("ab", 256);
        vocabulary.put("abc", 257);
        root.put("model", new LinkedHashMap<>(Map.of("type", "BPE", "vocab", vocabulary,
                "merges", List.of(List.of("a", "b"), List.of("ab", "c")))));
        root.put("added_tokens", added());
        return root;
    }

    private static List<Map<String, Object>> added() {
        List<Map<String, Object>> tokens = new ArrayList<>();
        String[] names = {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"};
        for (int index = 0; index < names.length; index++) {
            tokens.add(Map.of("id", 258 + index, "content", names[index], "special", index < 3,
                    "single_word", false, "lstrip", false, "rstrip", false, "normalized", false));
        }
        return tokens;
    }
}
