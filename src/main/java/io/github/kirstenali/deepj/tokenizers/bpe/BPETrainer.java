package io.github.kirstenali.deepj.tokenizers.bpe;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class BPETrainer {

    private static final byte[] END_OF_WORD = new byte[0];
    private static final String EOW_KEY = "<EOW_INTERNAL>";
    public static final List<String> DEFAULT_SPECIAL_TOKENS = List.of("<BOS>", "<EOS>", "<PAD>");

    public BPEModel train(String text, int targetVocabSize) {
        return train(text, targetVocabSize, List.of());
    }

    public BPEModel train(String text, int targetVocabSize, List<String> specialTokens) {
        List<String> normalizedSpecials = normalizeSpecialTokens(specialTokens);
        validateTargetVocabSize(targetVocabSize, normalizedSpecials.size());

        VocabularyState vocab = createBaseVocabulary();
        Map<String, Integer> specialTokenToId = addSpecialTokens(vocab, normalizedSpecials);
        List<List<Integer>> words = buildInitialWords(text, vocab.endOfWordId());

        List<TokenPair> merges = new ArrayList<>();
        Map<TokenPair, Integer> mergeToNewId = new HashMap<>();

        trainMerges(words, vocab, targetVocabSize, merges, mergeToNewId);

        return new BPEModel(
                vocab.idToBytes(),
                vocab.tokenKeyToId(),
                merges,
                mergeToNewId,
                vocab.endOfWordId(),
                BPEModel.CURRENT_FORMAT_VERSION,
                specialTokenToId
        );
    }

    private void trainMerges(
            List<List<Integer>> words,
            VocabularyState vocab,
            int targetVocabSize,
            List<TokenPair> merges,
            Map<TokenPair, Integer> mergeToNewId
    ) {
        while (vocab.size() < targetVocabSize) {
            TokenPair bestPair = findBestPair(words, vocab);
            if (bestPair == null) {
                return;
            }

            int newId = addMergedToken(vocab, bestPair);
            merges.add(bestPair);
            mergeToNewId.put(bestPair, newId);
            applyMerge(words, bestPair, newId);
        }
    }

    public BPEModel trainFromFile(Path path, int targetVocabSize) throws IOException {
        return train(Files.readString(path), targetVocabSize);
    }

    public BPEModel trainFromFile(Path path, int targetVocabSize, List<String> specialTokens) throws IOException {
        return train(Files.readString(path), targetVocabSize, specialTokens);
    }

    public BPETokenizer trainTokenizer(String text, int targetVocabSize) {
        return new BPETokenizer(train(text, targetVocabSize));
    }

    public BPETokenizer trainTokenizer(String text, int targetVocabSize, List<String> specialTokens) {
        return new BPETokenizer(train(text, targetVocabSize, specialTokens));
    }

    public BPETokenizer trainProductionTokenizer(String text, int targetVocabSize) {
        return trainTokenizer(text, targetVocabSize, DEFAULT_SPECIAL_TOKENS);
    }

    public BPETokenizer trainTokenizerFromFile(Path path, int targetVocabSize) throws IOException {
        return new BPETokenizer(trainFromFile(path, targetVocabSize));
    }

    public BPETokenizer trainTokenizerFromFile(Path path, int targetVocabSize, List<String> specialTokens) throws IOException {
        return new BPETokenizer(trainFromFile(path, targetVocabSize, specialTokens));
    }

    private void validateTargetVocabSize(int targetVocabSize, int specialCount) {
        int minimum = 257 + specialCount;
        if (targetVocabSize <= minimum) {
            throw new IllegalArgumentException("targetVocabSize must be > " + minimum);
        }
    }

    private List<String> normalizeSpecialTokens(List<String> specialTokens) {
        if (specialTokens == null || specialTokens.isEmpty()) {
            return List.of();
        }
        LinkedHashMap<String, Boolean> dedup = new LinkedHashMap<>();
        for (String token : specialTokens) {
            if (token == null || token.isEmpty()) {
                throw new IllegalArgumentException("special token cannot be null/empty");
            }
            dedup.put(token, Boolean.TRUE);
        }
        return List.copyOf(dedup.keySet());
    }

    private Map<String, Integer> addSpecialTokens(VocabularyState vocab, List<String> specialTokens) {
        if (specialTokens.isEmpty()) {
            return Map.of();
        }

        Map<String, Integer> specialTokenToId = new LinkedHashMap<>();
        for (String token : specialTokens) {
            String key = specialKey(token);
            if (vocab.contains(key)) {
                throw new IllegalArgumentException("duplicate special token: " + token);
            }
            int id = vocab.add(new byte[0], key);
            specialTokenToId.put(token, id);
        }
        return Map.copyOf(specialTokenToId);
    }

    private String specialKey(String token) {
        return "<SPECIAL_INTERNAL>:" + token;
    }

    private VocabularyState createBaseVocabulary() {
        List<byte[]> idToBytes = new ArrayList<>();
        Map<String, Integer> tokenKeyToId = new HashMap<>();

        for (int i = 0; i < 256; i++) {
            byte[] token = new byte[]{(byte) i};
            idToBytes.add(token);
            tokenKeyToId.put(BPEBytes.key(token), i);
        }

        int endOfWordId = idToBytes.size();
        idToBytes.add(END_OF_WORD);
        tokenKeyToId.put(EOW_KEY, endOfWordId);

        return new VocabularyState(idToBytes, tokenKeyToId, endOfWordId);
    }

    private List<List<Integer>> buildInitialWords(String text, int endOfWordId) {
        List<List<Integer>> words = new ArrayList<>();

        for (String piece : BPEBytes.splitPreserveWhitespace(text)) {
            words.add(toTokenIds(piece, endOfWordId));
        }

        return words;
    }

    static List<Integer> toTokenIds(String piece, int endOfWordId) {
        return BPEBytes.toTokenIds(piece, endOfWordId);
    }

    private TokenPair findBestPair(List<List<Integer>> words, VocabularyState vocab) {
        Map<TokenPair, Integer> counts = countPairs(words, vocab.endOfWordId());
        return selectBestValidPair(counts, vocab);
    }

    private Map<TokenPair, Integer> countPairs(List<List<Integer>> words, int endOfWordId) {
        Map<TokenPair, Integer> counts = new HashMap<>();

        for (List<Integer> word : words) {
            for (int i = 0; i < word.size() - 1; i++) {
                int left = word.get(i);
                int right = word.get(i + 1);

                if (right == endOfWordId) {
                    continue;
                }

                counts.merge(new TokenPair(left, right), 1, Integer::sum);
            }
        }

        return counts;
    }

    static TokenPair selectBestPair(Map<TokenPair, Integer> counts) {
        TokenPair bestPair = null;
        int bestCount = 1;

        for (Map.Entry<TokenPair, Integer> entry : counts.entrySet()) {
            TokenPair candidate = entry.getKey();
            int count = entry.getValue();
            if (count <= 1) {
                continue;
            }
            if (isBetterCandidate(candidate, count, bestPair, bestCount)) {
                bestPair = candidate;
                bestCount = count;
            }
        }

        return bestPair;
    }

    private TokenPair selectBestValidPair(Map<TokenPair, Integer> counts, VocabularyState vocab) {
        return chooseBestPair(counts.entrySet(), vocab);
    }

    private TokenPair chooseBestPair(Collection<Map.Entry<TokenPair, Integer>> entries, VocabularyState vocab) {
        TokenPair bestPair = null;
        int bestCount = 1;

        for (Map.Entry<TokenPair, Integer> entry : entries) {
            TokenPair candidate = entry.getKey();
            int count = entry.getValue();

            if (!isEligibleCandidate(vocab, candidate, count)) {
                continue;
            }

            if (isBetterCandidate(candidate, count, bestPair, bestCount)) {
                bestPair = candidate;
                bestCount = count;
            }
        }

        return bestPair;
    }

    private boolean isEligibleCandidate(VocabularyState vocab, TokenPair candidate, int count) {
        if (count <= 1) {
            return false;
        }
        return vocab == null || !wouldCreateExistingToken(vocab, candidate);
    }

    private static boolean isBetterCandidate(TokenPair candidate, int count, TokenPair bestPair, int bestCount) {
        if (count > bestCount) {
            return true;
        }
        return count == bestCount && (bestPair == null || candidate.compareTo(bestPair) < 0);
    }

    private boolean wouldCreateExistingToken(VocabularyState vocab, TokenPair pair) {
        return vocab.contains(BPEBytes.key(mergedBytes(vocab, pair)));
    }

    private int addMergedToken(VocabularyState vocab, TokenPair pair) {
        byte[] mergedBytes = mergedBytes(vocab, pair);
        return vocab.add(mergedBytes, BPEBytes.key(mergedBytes));
    }

    private byte[] mergedBytes(VocabularyState vocab, TokenPair pair) {
        return BPEBytes.concat(
                vocab.idToBytes().get(pair.left()),
                vocab.idToBytes().get(pair.right())
        );
    }

    private void applyMerge(List<List<Integer>> words, TokenPair pair, int newId) {
        for (List<Integer> word : words) {
            List<Integer> merged = BPEBytes.mergePair(word, pair, newId);
            word.clear();
            word.addAll(merged);
        }
    }
}