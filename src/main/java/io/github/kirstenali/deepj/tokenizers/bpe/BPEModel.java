package io.github.kirstenali.deepj.tokenizers.bpe;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public record BPEModel(
        List<byte[]> idToBytes,
        Map<String, Integer> tokenKeyToId,
        List<TokenPair> merges,
        Map<TokenPair, Integer> mergeToNewId,
        int endOfWordId,
        int modelFormatVersion,
        Map<String, Integer> specialTokenToId
) {

    public static final int CURRENT_FORMAT_VERSION = 1;

    public BPEModel(
            List<byte[]> idToBytes,
            Map<String, Integer> tokenKeyToId,
            List<TokenPair> merges,
            Map<TokenPair, Integer> mergeToNewId,
            int endOfWordId
    ) {
        this(idToBytes, tokenKeyToId, merges, mergeToNewId, endOfWordId, CURRENT_FORMAT_VERSION, Map.of());
    }

    public BPEModel(
            List<byte[]> idToBytes,
            Map<String, Integer> tokenKeyToId,
            List<TokenPair> merges,
            Map<TokenPair, Integer> mergeToNewId,
            int endOfWordId,
            int modelFormatVersion,
            Map<String, Integer> specialTokenToId
    ) {
        this.idToBytes = deepCopyBytes(idToBytes);
        this.tokenKeyToId = Map.copyOf(tokenKeyToId);
        this.merges = List.copyOf(merges);
        this.mergeToNewId = Map.copyOf(mergeToNewId);
        this.endOfWordId = endOfWordId;
        this.modelFormatVersion = modelFormatVersion;
        this.specialTokenToId = Map.copyOf(specialTokenToId);
    }

    @Override
    public List<byte[]> idToBytes() {
        return deepCopyBytes(idToBytes);
    }

    @Override
    public Map<String, Integer> specialTokenToId() {
        return Map.copyOf(specialTokenToId);
    }

    public int vocabSize() {
        return idToBytes.size();
    }

    public boolean hasSpecialTokens() {
        return !specialTokenToId.isEmpty();
    }

    public Map<Integer, String> idToSpecialToken() {
        Map<Integer, String> out = new HashMap<>(specialTokenToId.size());
        for (Map.Entry<String, Integer> e : specialTokenToId.entrySet()) {
            out.put(e.getValue(), e.getKey());
        }
        return Map.copyOf(out);
    }

    private static List<byte[]> deepCopyBytes(List<byte[]> source) {
        List<byte[]> copy = new ArrayList<>(source.size());
        for (byte[] token : source) {
            copy.add(Arrays.copyOf(token, token.length));
        }
        return List.copyOf(copy);
    }
}