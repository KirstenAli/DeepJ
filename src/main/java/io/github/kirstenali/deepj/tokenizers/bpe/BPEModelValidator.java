package io.github.kirstenali.deepj.tokenizers.bpe;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class BPEModelValidator {

    private BPEModelValidator() {}

    static void validate(List<byte[]> vocab, Map<String, Integer> tokenIds,
                         List<TokenPair> merges, Map<TokenPair, Integer> mergeIds,
                         int endOfWordId, int version, Map<String, Integer> specialIds) {
        requirePresent(vocab, tokenIds, merges, mergeIds, specialIds);
        requireVersion(version);
        requireValidVocab(vocab, endOfWordId);
        requireIdsInRange(tokenIds, vocab.size(), "token");
        requireValidMerges(vocab, merges, mergeIds);
        requireValidSpecials(specialIds, vocab.size(), endOfWordId);
    }

    private static void requirePresent(Object... values) {
        for (Object value : values) {
            if (value == null) throw invalid("Model fields must not be null");
        }
    }

    private static void requireVersion(int version) {
        if (version != BPEModel.CURRENT_FORMAT_VERSION) {
            throw invalid("Unsupported BPE model version: " + version);
        }
    }

    private static void requireValidVocab(List<byte[]> vocab, int endOfWordId) {
        if (vocab.isEmpty()) throw invalid("Vocabulary must not be empty");
        if (endOfWordId < 0 || endOfWordId >= vocab.size()) throw invalid("Invalid end-of-word id");
        for (byte[] token : vocab) {
            if (token == null) throw invalid("Vocabulary contains null token");
        }
        if (vocab.get(endOfWordId).length != 0) throw invalid("End-of-word token must be empty");
    }

    private static void requireIdsInRange(Map<?, Integer> ids, int vocabSize, String kind) {
        for (Integer id : ids.values()) {
            if (id == null || id < 0 || id >= vocabSize) throw invalid("Invalid " + kind + " id: " + id);
        }
    }

    private static void requireValidMerges(List<byte[]> vocab, List<TokenPair> merges,
                                           Map<TokenPair, Integer> mergeIds) {
        if (merges.size() != mergeIds.size()) throw invalid("Merge tables have different sizes");
        Set<TokenPair> pairs = new HashSet<>();
        Set<Integer> results = new HashSet<>();
        for (TokenPair pair : merges) {
            validateMerge(vocab, mergeIds, pairs, results, pair);
        }
    }

    private static void validateMerge(List<byte[]> vocab, Map<TokenPair, Integer> mergeIds,
                                      Set<TokenPair> pairs, Set<Integer> results, TokenPair pair) {
        requireUniquePair(pair, pairs);
        int result = requireMergeResult(pair, mergeIds, vocab.size());
        requireAvailableInputs(pair, result);
        requireUniqueResult(result, results);
    }

    private static void requireUniquePair(TokenPair pair, Set<TokenPair> pairs) {
        if (pair == null || !pairs.add(pair)) throw invalid("Duplicate or null merge pair");
    }

    private static int requireMergeResult(TokenPair pair, Map<TokenPair, Integer> ids, int size) {
        Integer result = ids.get(pair);
        if (result == null || result < 0 || result >= size) throw invalid("Invalid merge result id");
        return result;
    }

    private static void requireAvailableInputs(TokenPair pair, int result) {
        if (pair.left() < 0 || pair.right() < 0) throw unavailable(pair);
        if (pair.left() >= result || pair.right() >= result) throw unavailable(pair);
    }

    private static void requireUniqueResult(int result, Set<Integer> results) {
        if (!results.add(result)) throw invalid("Duplicate merge result id: " + result);
    }

    private static IllegalArgumentException unavailable(TokenPair pair) {
        return invalid("Merge references an unavailable token: " + pair);
    }

    private static void requireValidSpecials(Map<String, Integer> specials, int vocabSize, int endOfWordId) {
        requireIdsInRange(specials, vocabSize, "special token");
        Set<Integer> ids = new HashSet<>();
        for (Map.Entry<String, Integer> entry : specials.entrySet()) {
            if (entry.getKey() == null || entry.getKey().isEmpty()) throw invalid("Invalid special token");
            if (entry.getValue() == endOfWordId || !ids.add(entry.getValue())) {
                throw invalid("Duplicate or reserved special token id");
            }
        }
    }

    private static IllegalArgumentException invalid(String message) {
        return new IllegalArgumentException(message);
    }
}
