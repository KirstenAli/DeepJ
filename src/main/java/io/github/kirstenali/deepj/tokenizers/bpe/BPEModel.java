package io.github.kirstenali.deepj.tokenizers.bpe;

import java.util.List;
import java.util.Map;

public record BPEModel(
        List<byte[]> idToBytes,
        Map<String, Integer> tokenKeyToId,
        List<TokenPair> merges,
        Map<TokenPair, Integer> mergeToNewId,
        int endOfWordId
) {
    public BPEModel(
            List<byte[]> idToBytes,
            Map<String, Integer> tokenKeyToId,
            List<TokenPair> merges,
            Map<TokenPair, Integer> mergeToNewId,
            int endOfWordId
    ) {
        this.idToBytes = List.copyOf(idToBytes);
        this.tokenKeyToId = Map.copyOf(tokenKeyToId);
        this.merges = List.copyOf(merges);
        this.mergeToNewId = Map.copyOf(mergeToNewId);
        this.endOfWordId = endOfWordId;
    }

    public int vocabSize() {
        return idToBytes.size();
    }
}