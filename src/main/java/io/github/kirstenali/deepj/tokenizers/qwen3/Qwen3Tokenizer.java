package io.github.kirstenali.deepj.tokenizers.qwen3;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Hugging Face byte-level BPE, with the splitting and normalization rules read from tokenizer.json. */
public final class Qwen3Tokenizer implements Tokenizer {
    private final ByteAlphabet alphabet = new ByteAlphabet();
    private final TokenizerVocabulary vocabulary;
    private final TokenizerRecipe recipe;
    private final BytePairMerges merges;
    private final int end, endText;

    private Qwen3Tokenizer(Map<String, Object> root) {
        vocabulary = new TokenizerVocabulary(root);
        recipe = TokenizerRecipe.parse(root);
        merges = new BytePairMerges(Qwen3Json.array(Qwen3Json.object(root.get("model")).get("merges")),
                vocabulary.encoder);
        end = vocabulary.id("<|im_end|>");
        endText = vocabulary.id("<|endoftext|>");
        vocabulary.id("<|im_start|>");
    }

    public static Qwen3Tokenizer load(Path directory) throws IOException {
        try { return new Qwen3Tokenizer(Qwen3Json.read(directory.resolve("tokenizer.json"))); }
        catch (IllegalArgumentException | ArithmeticException failure) {
            throw new IOException("Unsupported or invalid Qwen3 tokenizer", failure);
        }
    }

    @Override public int vocabSize() { return vocabulary.size; }
    @Override public boolean isEndOfSequence(int id) { return id == end || id == endText; }

    @Override public int[] encode(String text) {
        Objects.requireNonNull(text, "text");
        List<Integer> result = new ArrayList<>();
        var matcher = vocabulary.addedPattern.matcher(text);
        int start = 0;
        while (matcher.find()) {
            encodeOrdinary(text.substring(start, matcher.start()), result);
            result.add(vocabulary.id(matcher.group()));
            start = matcher.end();
        }
        encodeOrdinary(text.substring(start), result);
        return result.stream().mapToInt(Integer::intValue).toArray();
    }

    private void encodeOrdinary(String text, List<Integer> output) {
        String normalized = recipe.normalize(text);
        var matcher = recipe.split().matcher(normalized);
        int end = 0;
        while (matcher.find()) {
            if (matcher.start() != end) throw new IllegalArgumentException("Tokenizer regex left unmatched text");
            merges.apply(alphabet.encode(matcher.group())).forEach(piece -> output.add(vocabulary.id(piece)));
            end = matcher.end();
        }
        if (end != normalized.length()) throw new IllegalArgumentException("Tokenizer regex left unmatched text");
    }

    @Override public String decode(int[] ids) { return decode(ids, false); }

    @Override public String decodePartial(int[] ids) { return decode(ids, false, true); }

    public String decode(int[] ids, boolean skipSpecialTokens) {
        return decode(ids, skipSpecialTokens, false);
    }

    private String decode(int[] ids, boolean skipSpecialTokens, boolean partial) {
        StringBuilder result = new StringBuilder();
        var bytes = new ByteArrayOutputStream();
        for (int id : ids) {
            String piece = vocabulary.decoder.get(id);
            if (piece == null) throw new IllegalArgumentException("Unknown tokenizer ID: " + id);
            if (vocabulary.added.contains(id)) {
                flush(bytes, result);
                if (!skipSpecialTokens || !vocabulary.special.contains(id)) result.append(piece);
            } else alphabet.append(piece, bytes);
        }
        if (partial) appendCompleteCharacters(bytes, result);
        else flush(bytes, result);
        return result.toString();
    }

    private static void appendCompleteCharacters(ByteArrayOutputStream bytes, StringBuilder result) {
        var input = ByteBuffer.wrap(bytes.toByteArray());
        var output = CharBuffer.allocate(input.remaining());
        var decoder = StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPLACE);
        // More tokens can finish a character whose bytes are still incomplete.
        decoder.decode(input, output, false);
        result.append(output.flip());
    }

    private static void flush(ByteArrayOutputStream bytes, StringBuilder result) {
        result.append(bytes.toString(StandardCharsets.UTF_8));
        bytes.reset();
    }
}
