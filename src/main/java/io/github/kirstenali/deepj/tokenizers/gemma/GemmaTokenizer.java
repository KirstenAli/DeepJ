package io.github.kirstenali.deepj.tokenizers.gemma;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/** Gemma's Unicode BPE with space replacement, special tokens, and UTF-8 byte fallback. */
public final class GemmaTokenizer implements Tokenizer {
    private final Map<String, Integer> vocabulary = new HashMap<>();
    private final Map<Pair, Integer> ranks = new HashMap<>();
    private final Map<Integer, String> decoder = new HashMap<>();
    private final Map<String, Integer> added = new HashMap<>();
    private final Pattern special;

    public GemmaTokenizer(Path directory) throws IOException {
        Map<String, Object> root = Qwen3Json.read(directory.resolve("tokenizer.json"));
        var model = Qwen3Json.object(root.get("model"));
        if (!"BPE".equals(model.get("type")) || !Boolean.TRUE.equals(model.get("byte_fallback"))) {
            throw new IOException("Expected Gemma byte-fallback BPE");
        }
        vocabulary(model);
        merges(Qwen3Json.array(model.get("merges")));
        for (Object item : Qwen3Json.array(root.get("added_tokens"))) {
            var token = Qwen3Json.object(item);
            added.put(Qwen3Json.string(token.get("content")), Math.toIntExact(Qwen3Json.integer(token.get("id"))));
        }
        special = Pattern.compile(added.keySet().stream().sorted(Comparator.comparingInt(String::length).reversed())
                .map(Pattern::quote).collect(java.util.stream.Collectors.joining("|")));
    }

    private void vocabulary(Map<String, Object> model) {
        Qwen3Json.object(model.get("vocab")).forEach((piece, number) -> {
            int id = Math.toIntExact(Qwen3Json.integer(number));
            vocabulary.put(piece, id);
            decoder.put(id, piece);
        });
    }

    private void merges(List<?> entries) {
        for (int rank = 0; rank < entries.size(); rank++) {
            var pair = mergePair(entries.get(rank));
            ranks.put(new Pair(Qwen3Json.string(pair.get(0)), Qwen3Json.string(pair.get(1))), rank);
        }
    }

    private static List<?> mergePair(Object value) {
        List<?> pair = value instanceof String text ? List.of(text.split(" ", 2)) : Qwen3Json.array(value);
        if (pair.size() != 2) throw new IllegalArgumentException("Invalid Gemma merge pair");
        return pair;
    }

    @Override public int vocabSize() { return vocabulary.size(); }
    @Override public boolean isEndOfSequence(int id) { return id == 1 || id == 107; }

    @Override public int[] encode(String text) {
        Objects.requireNonNull(text);
        List<Integer> output = new ArrayList<>();
        output.add(2);
        var matcher = special.matcher(text);
        int start = 0;
        while (matcher.find()) {
            ordinary(text.substring(start, matcher.start()), output);
            output.add(added.get(matcher.group()));
            start = matcher.end();
        }
        ordinary(text.substring(start), output);
        return output.stream().mapToInt(Integer::intValue).toArray();
    }

    private void ordinary(String text, List<Integer> output) {
        var pieces = new ArrayList<String>();
        text.replace(' ', '▁').codePoints().forEach(code -> addSymbol(new String(Character.toChars(code)), pieces));
        int position;
        while ((position = best(pieces)) >= 0) {
            pieces.set(position, pieces.get(position) + pieces.remove(position + 1));
        }
        pieces.forEach(piece -> output.add(vocabulary.getOrDefault(piece, 3)));
    }

    private void addSymbol(String symbol, List<String> pieces) {
        if (vocabulary.containsKey(symbol)) pieces.add(symbol);
        else {
            for (byte value : symbol.getBytes(StandardCharsets.UTF_8)) {
                pieces.add("<0x%02X>".formatted(Byte.toUnsignedInt(value)));
            }
        }
    }

    private int best(List<String> pieces) {
        int position = -1, minimum = Integer.MAX_VALUE;
        for (int i = 0; i + 1 < pieces.size(); i++) {
            Integer rank = ranks.get(new Pair(pieces.get(i), pieces.get(i + 1)));
            if (rank != null && rank < minimum) {
                minimum = rank;
                position = i;
            }
        }
        return position;
    }

    @Override public String decode(int[] tokens) {
        return decode(tokens, false);
    }

    @Override public String decodePartial(int[] tokens) { return decode(tokens, true); }

    private String decode(int[] tokens, boolean partial) {
        StringBuilder result = new StringBuilder();
        var bytes = new ByteArrayOutputStream();
        for (int token : tokens) {
            String piece = decoder.get(token);
            if (piece == null) throw new IllegalArgumentException("Unknown Gemma token: " + token);
            if (piece.matches("<0x[0-9A-F]{2}>")) bytes.write(Integer.parseInt(piece.substring(3, 5), 16));
            else {
                result.append(bytes.toString(StandardCharsets.UTF_8));
                bytes.reset();
                result.append(piece.replace('▁', ' '));
            }
        }
        return result.append(decodeBytes(bytes.toByteArray(), partial)).toString();
    }

    private static String decodeBytes(byte[] bytes, boolean partial) {
        if (!partial) return new String(bytes, StandardCharsets.UTF_8);
        var decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(java.nio.charset.CodingErrorAction.REPLACE);
        var output = java.nio.CharBuffer.allocate(bytes.length);
        decoder.decode(java.nio.ByteBuffer.wrap(bytes), output, false);
        return output.flip().toString();
    }

    private record Pair(String left, String right) {}
}
