package io.github.kirstenali.deepj.tokenizers.bpe;

import java.util.List;
import java.util.Map;

record VocabularyState(
        List<byte[]> idToBytes,
        Map<String, Integer> tokenKeyToId,
        int endOfWordId
) {
    int size() {
        return idToBytes.size();
    }

    boolean contains(String key) {
        return tokenKeyToId.containsKey(key);
    }

    int add(byte[] bytes, String key) {
        int newId = idToBytes.size();
        idToBytes.add(bytes);
        tokenKeyToId.put(key, newId);
        return newId;
    }
}