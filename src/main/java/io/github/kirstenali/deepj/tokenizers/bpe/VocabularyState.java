package io.github.kirstenali.deepj.tokenizers.bpe;

import java.util.List;
import java.util.Map;

record VocabularyState(
        List<byte[]> idToBytes,
        Map<String, Integer> bytesToId,
        int endOfWordId
) {
    int size() {
        return idToBytes.size();
    }

    boolean contains(String key) {
        return bytesToId.containsKey(key);
    }

    int add(byte[] bytes, String key) {
        int newId = idToBytes.size();
        idToBytes.add(bytes);
        bytesToId.put(key, newId);
        return newId;
    }
}