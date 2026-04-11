package io.github.kirstenali.deepj.tokenizers.bpe;

import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class BPETokenizer implements Tokenizer {

    private final BPEModel model;

    // Precomputed once at construction — model is immutable so these never change.
    private final int              endOfWordId;
    private final List<TokenPair>  merges;
    private final Map<TokenPair, Integer> mergeToNewId;
    private final List<SpecialTokenEntry> sortedSpecials;
    private final Map<Integer, String>    idToSpecial;

    public BPETokenizer(BPEModel model) {
        this.model        = model;
        this.endOfWordId  = model.endOfWordId();
        this.merges       = model.merges();
        this.mergeToNewId = model.mergeToNewId();

        SpecialTokenViews views = buildSpecialTokenViews(model.specialTokenToId());
        this.idToSpecial    = views.idToSpecial();
        this.sortedSpecials = views.sortedSpecials();
    }

    /** Builds both special-token views in a single pass to avoid iterating the map twice. */
    private static SpecialTokenViews buildSpecialTokenViews(Map<String, Integer> specials) {
        Map<Integer, String> inverse = new HashMap<>(specials.size());
        List<SpecialTokenEntry> sorted = new ArrayList<>(specials.size());

        for (Map.Entry<String, Integer> e : specials.entrySet()) {
            inverse.put(e.getValue(), e.getKey());
            sorted.add(new SpecialTokenEntry(e.getKey(), e.getValue()));
        }
        sorted.sort(Comparator.comparingInt((SpecialTokenEntry e) -> e.token().length()).reversed());

        return new SpecialTokenViews(Map.copyOf(inverse), List.copyOf(sorted));
    }


    public BPEModel model() {
        return model;
    }

    @Override
    public int[] encode(String text) {
        List<Integer> ids = new ArrayList<>();
        if (sortedSpecials.isEmpty()) {
            appendEncodedSegment(ids, text);
            return toIntArray(ids);
        }
        encodeWithSpecialTokens(text, ids);
        return toIntArray(ids);
    }

    private void encodeWithSpecialTokens(String text, List<Integer> ids) {
        int cursor = 0;
        while (cursor < text.length()) {
            SpecialTokenEntry matched = findSpecialTokenAt(text, cursor);
            if (matched != null) {
                ids.add(matched.id());
                cursor += matched.token().length();
                continue;
            }
            int nextSpecialIndex = nextSpecialStart(text, cursor);
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
        // idToBytesView() avoids the deep-copy done by idToBytes() — safe here because we only read.
        List<byte[]> vocab = model.idToBytesView();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        StringBuilder textOut = new StringBuilder();
        boolean touchedSpecial = false;

        for (int id : ids) {
            validateTokenId(id, vocab.size());
            touchedSpecial |= appendDecodedToken(id, vocab, out, textOut);
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
            ByteArrayOutputStream out,
            StringBuilder textOut
    ) {
        if (idToSpecial.containsKey(id)) {
            flushBytes(out, textOut);
            textOut.append(idToSpecial.get(id));
            return true;
        }

        if (id == endOfWordId) {
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
        List<Integer> tokens = BPEBytes.toTokenIds(piece, endOfWordId);

        for (TokenPair merge : merges) {
            Integer mergedId = mergeToNewId.get(merge);
            if (mergedId == null) {
                throw new IllegalStateException("merge table inconsistency: no id for " + merge);
            }
            tokens = BPEBytes.mergePair(tokens, merge, mergedId);
        }

        if (!tokens.isEmpty() && tokens.get(tokens.size() - 1) == endOfWordId) {
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

    private SpecialTokenEntry findSpecialTokenAt(String text, int index) {
        for (SpecialTokenEntry special : sortedSpecials) {
            if (text.startsWith(special.token(), index)) {
                return special;
            }
        }
        return null;
    }

    private int nextSpecialStart(String text, int cursor) {
        int next = text.length();
        for (SpecialTokenEntry special : sortedSpecials) {
            int idx = text.indexOf(special.token(), cursor);
            if (idx >= 0 && idx < next) {
                next = idx;
            }
        }
        return next;
    }

    private record SpecialTokenEntry(String token, int id) {}

    private record SpecialTokenViews(Map<Integer, String> idToSpecial, List<SpecialTokenEntry> sortedSpecials) {}

}