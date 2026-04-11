package io.github.kirstenali.deepj.tokenizers.bpe;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

public final class BPETrainer {

    private static final int    BASE_BYTE_VOCAB_SIZE     = 256;
    private static final int    BASE_VOCAB_SIZE_WITH_EOW = BASE_BYTE_VOCAB_SIZE + 1;
    private static final byte[] END_OF_WORD              = new byte[0];
    private static final String EOW_KEY                  = "<EOW_INTERNAL>";
    private static final String SPECIAL_KEY_PREFIX       = "<SPECIAL_INTERNAL>:";

    public static final List<String> DEFAULT_SPECIAL_TOKENS = List.of("<BOS>", "<EOS>", "<PAD>");

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    public BPEModel train(String text, int targetVocabSize) {
        return train(text, targetVocabSize, List.of());
    }

    public BPEModel train(String text, int targetVocabSize, List<String> specialTokens) {
        List<String> normalizedSpecials = normalizeSpecialTokens(specialTokens);
        validateTargetVocabSize(targetVocabSize, normalizedSpecials.size());

        VocabularyState      vocab            = createBaseVocabulary();
        Map<String, Integer> specialTokenToId = addSpecialTokens(vocab, normalizedSpecials);
        List<List<Integer>>  words            = buildInitialWords(text, vocab.endOfWordId());

        MergeResult result = trainMerges(words, vocab, targetVocabSize);

        return new BPEModel(
                vocab.idToBytes(),
                vocab.tokenKeyToId(),
                result.merges(),
                result.mergeToNewId(),
                vocab.endOfWordId(),
                BPEModel.CURRENT_FORMAT_VERSION,
                specialTokenToId
        );
    }

    @SuppressWarnings("unused") // public API — callers outside this module
    public BPEModel trainFromFile(Path path, int targetVocabSize) throws IOException {
        return train(Files.readString(path), targetVocabSize);
    }

    @SuppressWarnings("unused") // public API — callers outside this module
    public BPEModel trainFromFile(Path path, int targetVocabSize, List<String> specialTokens) throws IOException {
        return train(Files.readString(path), targetVocabSize, specialTokens);
    }

    public BPETokenizer trainTokenizer(String text, int targetVocabSize) {
        return new BPETokenizer(train(text, targetVocabSize));
    }

    public BPETokenizer trainTokenizer(String text, int targetVocabSize, List<String> specialTokens) {
        return new BPETokenizer(train(text, targetVocabSize, specialTokens));
    }

    public BPETokenizer trainTokenizerWithDefaults(String text, int targetVocabSize) {
        return trainTokenizer(text, targetVocabSize, DEFAULT_SPECIAL_TOKENS);
    }

    @SuppressWarnings("unused") // public API — callers outside this module
    public BPETokenizer trainTokenizerFromFile(Path path, int targetVocabSize) throws IOException {
        return trainTokenizerFromFile(path, targetVocabSize, List.of());
    }

    @SuppressWarnings("unused") // public API — callers outside this module
    public BPETokenizer trainTokenizerFromFile(Path path, int targetVocabSize, List<String> specialTokens) throws IOException {
        return new BPETokenizer(trainFromFile(path, targetVocabSize, specialTokens));
    }

    // -------------------------------------------------------------------------
    // Validation & normalisation
    // -------------------------------------------------------------------------

    private static void validateTargetVocabSize(int targetVocabSize, int specialCount) {
        int minimum = BASE_VOCAB_SIZE_WITH_EOW + specialCount;
        if (targetVocabSize <= minimum) {
            throw new IllegalArgumentException("targetVocabSize must be > " + minimum);
        }
    }

    private static List<String> normalizeSpecialTokens(List<String> specialTokens) {
        if (specialTokens == null || specialTokens.isEmpty()) {
            return List.of();
        }
        LinkedHashSet<String> dedup = new LinkedHashSet<>();
        for (String token : specialTokens) {
            if (token == null || token.isEmpty()) {
                throw new IllegalArgumentException("special token cannot be null/empty");
            }
            dedup.add(token);
        }
        return List.copyOf(dedup);
    }

    // -------------------------------------------------------------------------
    // Vocabulary construction
    // -------------------------------------------------------------------------

    private static VocabularyState createBaseVocabulary() {
        List<byte[]>         idToBytes    = new ArrayList<>();
        Map<String, Integer> tokenKeyToId = new HashMap<>();

        buildByteTokens(idToBytes, tokenKeyToId);
        int endOfWordId = registerEndOfWord(idToBytes, tokenKeyToId);

        return new VocabularyState(idToBytes, tokenKeyToId, endOfWordId);
    }

    private static void buildByteTokens(List<byte[]> idToBytes, Map<String, Integer> tokenKeyToId) {
        for (int i = 0; i < BASE_BYTE_VOCAB_SIZE; i++) {
            byte[] token = new byte[]{(byte) i};
            idToBytes.add(token);
            tokenKeyToId.put(BPEBytes.key(token), i);
        }
    }

    private static int registerEndOfWord(List<byte[]> idToBytes, Map<String, Integer> tokenKeyToId) {
        int endOfWordId = idToBytes.size();
        idToBytes.add(END_OF_WORD);
        tokenKeyToId.put(EOW_KEY, endOfWordId);
        return endOfWordId;
    }

    private static Map<String, Integer> addSpecialTokens(VocabularyState vocab, List<String> specialTokens) {
        if (specialTokens.isEmpty()) {
            return Map.of();
        }
        Map<String, Integer> specialTokenToId = new LinkedHashMap<>();
        for (String token : specialTokens) {
            String key = specialKey(token);
            if (vocab.contains(key)) {
                throw new IllegalArgumentException("duplicate special token: " + token);
            }
            int id = vocab.add(END_OF_WORD, key);
            specialTokenToId.put(token, id);
        }
        return Map.copyOf(specialTokenToId);
    }

    private static String specialKey(String token) {
        return SPECIAL_KEY_PREFIX + token;
    }

    // -------------------------------------------------------------------------
    // Word segmentation
    // -------------------------------------------------------------------------

    private static List<List<Integer>> buildInitialWords(String text, int endOfWordId) {
        List<List<Integer>> words = new ArrayList<>();
        for (String piece : BPEBytes.splitPreserveWhitespace(text)) {
            words.add(BPEBytes.toTokenIds(piece, endOfWordId));
        }
        return words;
    }

    // -------------------------------------------------------------------------
    // Merge training loop
    // -------------------------------------------------------------------------

    private static MergeResult trainMerges(List<List<Integer>> words, VocabularyState vocab, int targetVocabSize) {
        List<TokenPair>         merges       = new ArrayList<>();
        Map<TokenPair, Integer> mergeToNewId = new HashMap<>();
        Map<TokenPair, Integer> counts       = countPairs(words, vocab.endOfWordId());

        while (vocab.size() < targetVocabSize) {
            SelectedPair best = selectBestValidPair(counts, vocab);
            if (best == null) {
                break;
            }

            TokenPair pair  = best.pair();
            byte[]    bytes = best.merged();
            int       newId = vocab.add(bytes, BPEBytes.key(bytes));
            merges.add(pair);
            mergeToNewId.put(pair, newId);
            applyMerge(words, pair, newId, vocab.endOfWordId(), counts);
        }

        return new MergeResult(merges, mergeToNewId);
    }


    // -------------------------------------------------------------------------
    // Pair counting
    // -------------------------------------------------------------------------

    private static Map<TokenPair, Integer> countPairs(List<List<Integer>> words, int endOfWordId) {
        Map<TokenPair, Integer> counts = new HashMap<>();
        for (List<Integer> word : words) {
            countPairsInWord(word, endOfWordId, counts);
        }
        return counts;
    }

    private static void countPairsInWord(List<Integer> word, int endOfWordId, Map<TokenPair, Integer> counts) {
        for (int i = 0; i < word.size() - 1; i++) {
            int left  = word.get(i);
            int right = word.get(i + 1);
            if (right == endOfWordId) {
                continue;
            }
            counts.merge(new TokenPair(left, right), 1, Integer::sum);
        }
    }

    // -------------------------------------------------------------------------
    // Pair selection
    // -------------------------------------------------------------------------

    // @VisibleForTesting
    static TokenPair selectBestPair(Map<TokenPair, Integer> counts) {
        return chooseBestPair(counts, (candidate, count) -> count > 1);
    }

    private static SelectedPair selectBestValidPair(Map<TokenPair, Integer> counts, VocabularyState vocab) {
        TokenPair best = chooseBestPair(counts,
                (candidate, count) -> count > 1 && !vocab.contains(BPEBytes.key(mergedBytes(vocab, candidate))));
        return best == null ? null : new SelectedPair(best, mergedBytes(vocab, best));
    }

    private static TokenPair chooseBestPair(Map<TokenPair, Integer> counts, PairEligibility eligibility) {
        TokenPair bestPair  = null;
        int       bestCount = 1;

        for (Map.Entry<TokenPair, Integer> entry : counts.entrySet()) {
            TokenPair candidate = entry.getKey();
            int       count     = entry.getValue();

            if (!eligibility.accept(candidate, count)) {
                continue;
            }
            if (isBetterCandidate(candidate, count, bestPair, bestCount)) {
                bestPair  = candidate;
                bestCount = count;
            }
        }

        return bestPair;
    }

    private static boolean isBetterCandidate(TokenPair candidate, int count, TokenPair bestPair, int bestCount) {
        if (count > bestCount) {
            return true;
        }
        return count == bestCount && (bestPair == null || candidate.compareTo(bestPair) < 0);
    }

    // -------------------------------------------------------------------------
    // Merge application
    // -------------------------------------------------------------------------

    private static byte[] mergedBytes(VocabularyState vocab, TokenPair pair) {
        return BPEBytes.concat(
                vocab.idToBytes().get(pair.left()),
                vocab.idToBytes().get(pair.right())
        );
    }

    private static void applyMerge(List<List<Integer>> words, TokenPair pair, int newId,
                                   int endOfWordId, Map<TokenPair, Integer> counts) {
        for (List<Integer> word : words) {
            applyMergeInPlace(word, pair, newId, endOfWordId, counts);
        }
    }

    /**
     * Two-pointer in-place merge. As each occurrence of {@code pair} is replaced by
     * {@code newId}, the counts for the neighboring pairs are updated incrementally:
     * <pre>
     *   (leftNeighbor, pair.left)  → (leftNeighbor, newId)
     *   (pair.right, rightNeighbor) → (newId, rightNeighbor)
     * </pre>
     * This keeps the counts map accurate across rounds without a full rescan.
     */
    private static void applyMergeInPlace(List<Integer> word, TokenPair pair, int newId,
                                          int endOfWordId, Map<TokenPair, Integer> counts) {
        int write = 0;
        int read  = 0;
        int size  = word.size();

        while (read < size) {
            boolean matched = read < size - 1
                    && word.get(read)     == pair.left()
                    && word.get(read + 1) == pair.right();

            if (matched) {
                if (write > 0) {
                    int left = word.get(write - 1);
                    adjustCount(counts, new TokenPair(left, pair.left()),  -1);
                    adjustCount(counts, new TokenPair(left, newId),        +1);
                }
                int rightRead = read + 2;
                if (rightRead < size) {
                    int right = word.get(rightRead);
                    if (right != endOfWordId) {
                        adjustCount(counts, new TokenPair(pair.right(), right), -1);
                        adjustCount(counts, new TokenPair(newId,        right), +1);
                    }
                }
                adjustCount(counts, pair, -1);
                word.set(write++, newId);
                read += 2;
            } else {
                word.set(write++, word.get(read++));
            }
        }

        if (write < size) {
            word.subList(write, size).clear();
        }
    }

    private static void adjustCount(Map<TokenPair, Integer> counts, TokenPair pair, int delta) {
        int updated = counts.getOrDefault(pair, 0) + delta;
        if (updated <= 0) {
            counts.remove(pair);
        } else {
            counts.put(pair, updated);
        }
    }

    // -------------------------------------------------------------------------
    // Inner types
    // -------------------------------------------------------------------------

    private record MergeResult(List<TokenPair> merges, Map<TokenPair, Integer> mergeToNewId) {}

    private record SelectedPair(TokenPair pair, byte[] merged) {}

    @FunctionalInterface
    private interface PairEligibility {
        boolean accept(TokenPair candidate, int count);
    }
}

