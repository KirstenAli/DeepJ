package io.github.kirstenali.deepj.tokenizers.bpe;

import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class BPETokenizer implements Tokenizer {

    private static final List<String> END_OF_SEQUENCE_TOKENS =
            List.of("<EOS>", "<|endoftext|>");
    private final BPEModel model;

    private final int              endOfWordId;
    private final BPEMergeTable mergeTable;
    private final List<SpecialTokenEntry> sortedSpecials;
    private final Map<Integer, String>    idToSpecial;
    private final Set<Integer>            endOfSequenceIds;

    public BPETokenizer(BPEModel model) {
        this.model        = model;
        this.endOfWordId  = model.endOfWordId();
        this.mergeTable   = new BPEMergeTable(model.merges(), model.mergeToNewId());

        SpecialTokenViews views = buildSpecialTokenViews(model.specialTokenToId());
        this.idToSpecial    = views.idToSpecial();
        this.sortedSpecials = views.sortedSpecials();
        this.endOfSequenceIds = endOfSequenceIds(model.specialTokenToId());
    }

    private static Set<Integer> endOfSequenceIds(Map<String, Integer> specials) {
        Set<Integer> ids = new java.util.HashSet<>();
        for (String token : END_OF_SEQUENCE_TOKENS) {
            Integer id = specials.get(token);
            if (id != null) ids.add(id);
        }
        return Set.copyOf(ids);
    }

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
        IntArrayBuilder ids = new IntArrayBuilder(text.length());
        if (sortedSpecials.isEmpty()) {
            appendEncodedSegment(ids, text, 0, text.length());
            return ids.toArray();
        }
        encodeWithSpecialTokens(text, ids);
        return ids.toArray();
    }

    private void encodeWithSpecialTokens(String text, IntArrayBuilder ids) {
        int cursor = 0;
        while (cursor < text.length()) {
            SpecialTokenEntry matched = findSpecialTokenAt(text, cursor);
            if (matched != null) {
                ids.add(matched.id());
                cursor += matched.token().length();
                continue;
            }
            int nextSpecialIndex = nextSpecialStart(text, cursor);
            appendEncodedSegment(ids, text, cursor, nextSpecialIndex);
            cursor = nextSpecialIndex;
        }
    }

    private void appendEncodedSegment(IntArrayBuilder ids, String text, int start, int end) {
        int pieceStart = start;
        while (pieceStart < end) {
            int pieceEnd = findPieceEnd(text, pieceStart, end);
            appendEncodedPiece(ids, text.substring(pieceStart, pieceEnd));
            pieceStart = pieceEnd;
        }
    }

    private static int findPieceEnd(String text, int start, int end) {
        boolean whitespace = Character.isWhitespace(text.charAt(start));
        int cursor = start + 1;
        while (cursor < end && Character.isWhitespace(text.charAt(cursor)) == whitespace) {
            cursor++;
        }
        return cursor;
    }

    @Override
    public String decode(int[] ids) {

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

    @Override
    public boolean isEndOfSequence(int tokenId) {
        return endOfSequenceIds.contains(tokenId);
    }

    private void appendEncodedPiece(IntArrayBuilder output, String piece) {
        int[] tokens = BPEBytes.toTokenArray(piece, endOfWordId);
        int size = applyMerges(tokens);
        if (size > 0 && tokens[size - 1] == endOfWordId) size--;
        output.addAll(tokens, size);
    }

    private int applyMerges(int[] tokens) {
        int size = tokens.length;
        int rank;
        while ((rank = bestRank(tokens, size)) != BPEMergeTable.NO_MERGE) {
            size = mergeInPlace(tokens, size, rank);
        }
        return size;
    }

    private int bestRank(int[] tokens, int size) {
        int best = Integer.MAX_VALUE;
        for (int i = 0; i < size - 1; i++) {
            int rank = mergeTable.rank(tokens[i], tokens[i + 1]);
            if (rank != BPEMergeTable.NO_MERGE && rank < best) best = rank;
        }
        return best == Integer.MAX_VALUE ? BPEMergeTable.NO_MERGE : best;
    }

    private int mergeInPlace(int[] tokens, int size, int rank) {
        int read = 0;
        int write = 0;
        while (read < size) {
            if (matches(tokens, read, size, rank)) {
                tokens[write++] = mergeTable.result(rank);
                read += 2;
            } else {
                tokens[write++] = tokens[read++];
            }
        }
        return write;
    }

    private boolean matches(int[] tokens, int index, int size, int rank) {
        return index + 1 < size
                && tokens[index] == mergeTable.left(rank)
                && tokens[index + 1] == mergeTable.right(rank);
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
