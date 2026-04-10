package io.github.kirstenali.deepj.tokenizers.bpe;

import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

public record BPETokenizer(BPEModel model) implements Tokenizer {

    @Override
    public int[] encode(String text) {
        List<Integer> ids = new ArrayList<>();
        if (!model.hasSpecialTokens()) {
            appendEncodedSegment(ids, text);
            return toIntArray(ids);
        }

        List<SpecialTokenEntry> specialsByLength = sortedSpecialTokensByLength();
        encodeWithSpecialTokens(text, specialsByLength, ids);
        return toIntArray(ids);
    }

    private void encodeWithSpecialTokens(String text, List<SpecialTokenEntry> specialsByLength, List<Integer> ids) {
        int cursor = 0;
        while (cursor < text.length()) {
            SpecialTokenEntry matched = findSpecialTokenAt(text, cursor, specialsByLength);
            if (matched != null) {
                ids.add(matched.id());
                cursor += matched.token().length();
                continue;
            }

            int nextSpecialIndex = nextSpecialStart(text, cursor, specialsByLength);
            appendEncodedSegment(ids, text.substring(cursor, nextSpecialIndex));
            cursor = nextSpecialIndex;
        }
    }

    private void appendEncodedSegment(List<Integer> ids, String segment) {
        for (String piece : BPEBytes.splitPreserveWhitespace(segment)) {
            ids.addAll(encodePiece(piece));
        }
    }

    private static int[] toIntArray(List<Integer> ids) {
        return ids.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public String decode(int[] ids) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        StringBuilder textOut = new StringBuilder();
        boolean touchedSpecial = false;
        List<byte[]> vocab = model.idToBytes();
        Map<Integer, String> idToSpecial = model.idToSpecialToken();

        for (int id : ids) {
            validateTokenId(id, vocab.size());
            touchedSpecial |= appendDecodedToken(id, vocab, idToSpecial, out, textOut);
        }

        if (!touchedSpecial) {
            return out.toString(StandardCharsets.UTF_8);
        }
        flushBytes(out, textOut);
        return textOut.toString();
    }

    private static void validateTokenId(int id, int vocabSize) {
        if (id < 0 || id >= vocabSize) {
            throw new IllegalArgumentException("Token id out of range: " + id);
        }
    }

    private boolean appendDecodedToken(
            int id,
            List<byte[]> vocab,
            Map<Integer, String> idToSpecial,
            ByteArrayOutputStream out,
            StringBuilder textOut
    ) {
        if (idToSpecial.containsKey(id)) {
            flushBytes(out, textOut);
            textOut.append(idToSpecial.get(id));
            return true;
        }

        if (id == model.endOfWordId()) {
            return false;
        }

        byte[] bytes = vocab.get(id);
        out.write(bytes, 0, bytes.length);
        return false;
    }

    @Override
    public int vocabSize() {
        return model.vocabSize();
    }

    private List<Integer> encodePiece(String piece) {
        List<Integer> tokens = BPEBytes.toTokenIds(piece, model.endOfWordId());

        for (TokenPair merge : model.merges()) {
            int mergedId = model.mergeToNewId().get(merge);
            tokens = BPEBytes.mergePair(tokens, merge, mergedId);
        }

        if (!tokens.isEmpty() && tokens.get(tokens.size() - 1) == model.endOfWordId()) {
            tokens.remove(tokens.size() - 1);
        }

        return tokens;
    }

    private static void flushBytes(ByteArrayOutputStream out, StringBuilder textOut) {
        if (out.size() == 0) {
            return;
        }
        textOut.append(out.toString(StandardCharsets.UTF_8));
        out.reset();
    }

    private List<SpecialTokenEntry> sortedSpecialTokensByLength() {
        List<SpecialTokenEntry> entries = new ArrayList<>();
        for (Map.Entry<String, Integer> e : model.specialTokenToId().entrySet()) {
            entries.add(new SpecialTokenEntry(e.getKey(), e.getValue()));
        }
        entries.sort(Comparator.comparingInt((SpecialTokenEntry e) -> e.token().length()).reversed());
        return entries;
    }

    private static SpecialTokenEntry findSpecialTokenAt(String text, int index, List<SpecialTokenEntry> specials) {
        for (SpecialTokenEntry special : specials) {
            if (text.startsWith(special.token(), index)) {
                return special;
            }
        }
        return null;
    }

    private static int nextSpecialStart(String text, int cursor, List<SpecialTokenEntry> specials) {
        int next = text.length();
        for (SpecialTokenEntry special : specials) {
            int idx = text.indexOf(special.token(), cursor);
            if (idx >= 0 && idx < next) {
                next = idx;
            }
        }
        return next;
    }

    private record SpecialTokenEntry(String token, int id) {
    }

}