package io.github.kirstenali.deepj.tokenizers.qwen3;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.text.Normalizer;
import java.util.Map;
import java.util.regex.Pattern;

record TokenizerRecipe(Pattern split, boolean nfc) {
    static TokenizerRecipe parse(Map<String, Object> root) {
        Map<String, Object> sequence = Qwen3Json.object(root.get("pre_tokenizer"));
        requireType(sequence, "Sequence");
        var pieces = Qwen3Json.array(sequence.get("pretokenizers"));
        if (pieces.size() != 2) throw new IllegalArgumentException("Expected Split then ByteLevel");
        String regex = split(Qwen3Json.object(pieces.get(0)));
        byteLevel(Qwen3Json.object(pieces.get(1)));
        if (root.get("post_processor") != null) {
            requireType(Qwen3Json.object(root.get("post_processor")), "ByteLevel");
        }
        requireType(Qwen3Json.object(root.get("decoder")), "ByteLevel");
        Pattern pattern = Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
        return new TokenizerRecipe(pattern, nfc(root.get("normalizer")));
    }

    private static String split(Map<String, Object> split) {
        requireType(split, "Split");
        if (!"Isolated".equals(split.get("behavior")) || Boolean.TRUE.equals(split.get("invert"))) {
            throw new IllegalArgumentException("Unsupported tokenizer splitting behavior");
        }
        return Qwen3Json.string(Qwen3Json.object(split.get("pattern")).get("Regex"));
    }

    private static void byteLevel(Map<String, Object> value) {
        requireType(value, "ByteLevel");
        if (Boolean.TRUE.equals(value.get("add_prefix_space")) || Boolean.TRUE.equals(value.get("use_regex"))) {
            throw new IllegalArgumentException("Unsupported ByteLevel options");
        }
    }

    private static boolean nfc(Object value) {
        if (value == null) return false;
        Map<String, Object> normalization = Qwen3Json.object(value);
        if ("Sequence".equals(normalization.get("type"))) {
            return Qwen3Json.array(normalization.get("normalizers")).stream().map(TokenizerRecipe::nfc)
                    .reduce(false, (left, right) -> left || right);
        }
        requireType(normalization, "NFC");
        return true;
    }

    static void requireType(Map<String, Object> value, String type) {
        if (!type.equals(value.get("type"))) throw new IllegalArgumentException("Expected tokenizer type " + type);
    }

    String normalize(String text) { return nfc ? Normalizer.normalize(text, Normalizer.Form.NFC) : text; }
}
