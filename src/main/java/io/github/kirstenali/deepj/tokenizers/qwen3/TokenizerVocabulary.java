package io.github.kirstenali.deepj.tokenizers.qwen3;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

final class TokenizerVocabulary {
    final Map<String, Integer> encoder = new HashMap<>();
    final Map<Integer, String> decoder = new HashMap<>();
    final Set<Integer> added = new HashSet<>();
    final Set<Integer> special = new HashSet<>();
    final Pattern addedPattern;
    final int size;

    TokenizerVocabulary(Map<String, Object> root) {
        var model = Qwen3Json.object(root.get("model"));
        TokenizerRecipe.requireType(model, "BPE");
        requireBpeOptions(model);
        Qwen3Json.object(model.get("vocab")).forEach((piece, id) -> put(piece, Qwen3Json.integer(id)));
        requireBytes();
        for (Object item : Qwen3Json.array(root.get("added_tokens"))) {
            add(Qwen3Json.object(item));
        }
        addedPattern = Pattern.compile(added.stream().map(decoder::get)
                .sorted(java.util.Comparator.comparingInt(String::length).reversed())
                .map(Pattern::quote).collect(java.util.stream.Collectors.joining("|")));
        size = decoder.keySet().stream().mapToInt(Integer::intValue).max().orElseThrow() + 1;
    }

    private static void requireBpeOptions(Map<String, Object> model) {
        if (model.get("unk_token") != null || model.get("dropout") != null
                || Boolean.TRUE.equals(model.get("byte_fallback")) || Boolean.TRUE.equals(model.get("ignore_merges"))) {
            throw new IllegalArgumentException("Unsupported BPE options");
        }
        for (String name : new String[]{"continuing_subword_prefix", "end_of_word_suffix"}) {
            if (model.get(name) != null && !"".equals(model.get(name))) {
                throw new IllegalArgumentException("Unsupported BPE affix");
            }
        }
    }

    private void put(String piece, long id) {
        if (piece.isEmpty() || id < 0 || id >= 1_000_000) {
            throw new IllegalArgumentException("Invalid vocabulary entry");
        }
        Integer previous = encoder.putIfAbsent(piece, (int) id);
        String previousText = decoder.putIfAbsent((int) id, piece);
        if (previous != null && previous != id || previousText != null && !previousText.equals(piece)) {
            throw new IllegalArgumentException("Duplicate vocabulary ID or text");
        }
    }

    private void requireBytes() {
        var alphabet = new ByteAlphabet();
        for (int value = 0; value < 256; value++) {
            if (!encoder.containsKey(alphabet.symbol(value))) throw new IllegalArgumentException("Missing BPE byte");
        }
    }

    private void add(Map<String, Object> token) {
        if (Boolean.TRUE.equals(token.get("single_word")) || Boolean.TRUE.equals(token.get("lstrip"))
                || Boolean.TRUE.equals(token.get("rstrip")) || Boolean.TRUE.equals(token.get("normalized"))) {
            throw new IllegalArgumentException("Unsupported added-token behavior");
        }
        int id = Qwen3Json.integer(token, "id");
        put(Qwen3Json.string(token.get("content")), id);
        added.add(id);
        if (Boolean.TRUE.equals(token.get("special"))) special.add(id);
    }

    int id(String text) {
        Integer id = encoder.get(text);
        if (id == null) throw new IllegalArgumentException("Token missing from vocabulary: " + text);
        return id;
    }
}
