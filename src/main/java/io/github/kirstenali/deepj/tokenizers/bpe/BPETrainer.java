package io.github.kirstenali.deepj.tokenizers.bpe;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public final class BPETrainer {

    private static final int    BASE_BYTE_VOCAB_SIZE     = 256;
    private static final int    BASE_VOCAB_SIZE_WITH_EOW = BASE_BYTE_VOCAB_SIZE + 1;
    private static final byte[] END_OF_WORD              = new byte[0];
    private static final String EOW_KEY                  = "<EOW_INTERNAL>";
    private static final String SPECIAL_KEY_PREFIX       = "<SPECIAL_INTERNAL>:";

    private static final int FILE_SAMPLE_CHARS = 50 * 1024 * 1024;

    public static final List<String> DEFAULT_SPECIAL_TOKENS = List.of("<BOS>", "<EOS>", "<PAD>");

    public BPEModel train(String text, int targetVocabSize) {
        return train(text, targetVocabSize, List.of());
    }

    public BPEModel train(String text, int targetVocabSize, List<String> specialTokens) {
        List<String> normalizedSpecials = normalizeSpecialTokens(specialTokens);
        validateTargetVocabSize(targetVocabSize, normalizedSpecials.size());

        VocabularyState      vocab            = createBaseVocabulary();
        Map<String, Integer> specialTokenToId = addSpecialTokens(vocab, normalizedSpecials);
        List<TrainingWord>   words            = buildInitialWords(text, vocab.endOfWordId());

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

    public BPEModel trainFromFile(Path path, int targetVocabSize, List<String> specialTokens) throws IOException {
        return trainFromFile(path, targetVocabSize, specialTokens, FILE_SAMPLE_CHARS);
    }

    public BPEModel trainFromFile(Path path, int targetVocabSize, List<String> specialTokens,
                                  int sampleChars) throws IOException {
        if (sampleChars <= 0) throw new IllegalArgumentException("sampleChars must be > 0");
        return train(readSample(path, sampleChars), targetVocabSize, specialTokens);
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

    public BPETokenizer trainTokenizerWithDefaultsFromFile(Path path, int targetVocabSize) throws IOException {
        return trainTokenizerFromFile(path, targetVocabSize, DEFAULT_SPECIAL_TOKENS);
    }

    public BPETokenizer trainTokenizerFromFile(Path path, int targetVocabSize) throws IOException {
        return trainTokenizerFromFile(path, targetVocabSize, List.of());
    }

    public BPETokenizer trainTokenizerFromFile(Path path, int targetVocabSize, List<String> specialTokens) throws IOException {
        return new BPETokenizer(trainFromFile(path, targetVocabSize, specialTokens));
    }

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

    private static List<TrainingWord> buildInitialWords(String text, int endOfWordId) {
        Map<String, Integer> frequencies = countPieces(text);
        List<TrainingWord> words = new ArrayList<>(frequencies.size());
        frequencies.forEach((piece, count) -> words.add(trainingWord(piece, endOfWordId, count)));
        return words;
    }

    private static Map<String, Integer> countPieces(String text) {
        Map<String, Integer> frequencies = new LinkedHashMap<>();
        int start = 0;
        while (start < text.length()) {
            int end = endOfRun(text, start);
            frequencies.merge(text.substring(start, end), 1, Integer::sum);
            start = end;
        }
        return frequencies;
    }

    private static int endOfRun(String text, int start) {
        boolean whitespace = Character.isWhitespace(text.charAt(start));
        int end = start + 1;
        while (end < text.length() && Character.isWhitespace(text.charAt(end)) == whitespace) end++;
        return end;
    }

    private static TrainingWord trainingWord(String piece, int endOfWordId, int frequency) {
        return new TrainingWord(BPEBytes.toTokenArray(piece, endOfWordId), frequency);
    }

    private static MergeResult trainMerges(List<TrainingWord> words, VocabularyState vocab, int targetVocabSize) {
        List<TokenPair>         merges       = new ArrayList<>();
        Map<TokenPair, Integer> mergeToNewId = new HashMap<>();
        Map<TokenPair, Integer> counts       = countPairs(words, vocab.endOfWordId());
        PriorityQueue<PairEntry> queue       = buildQueue(counts);

        while (vocab.size() < targetVocabSize) {
            SelectedPair best = pollBestValidPair(queue, counts, vocab);
            if (best == null) break;

            TokenPair pair  = best.pair();
            byte[]    bytes = best.merged();
            int       newId = vocab.add(bytes, BPEBytes.key(bytes));
            merges.add(pair);
            mergeToNewId.put(pair, newId);
            applyMerge(words, pair, newId, vocab.endOfWordId(), counts, queue);
        }

        return new MergeResult(merges, mergeToNewId);
    }

    private static PriorityQueue<PairEntry> buildQueue(Map<TokenPair, Integer> counts) {
        PriorityQueue<PairEntry> queue = new PriorityQueue<>(Math.max(1, counts.size()));
        for (Map.Entry<TokenPair, Integer> e : counts.entrySet()) {
            if (e.getValue() > 1) queue.offer(new PairEntry(e.getValue(), e.getKey()));
        }
        return queue;
    }

    private static SelectedPair pollBestValidPair(PriorityQueue<PairEntry> queue,
                                                  Map<TokenPair, Integer> counts,
                                                  VocabularyState vocab) {
        while (!queue.isEmpty()) {
            PairEntry entry   = queue.poll();
            Integer   current = counts.get(entry.pair());
            if (current == null || current != entry.count() || current <= 1) continue;
            byte[] merged = mergedBytes(vocab, entry.pair());
            if (vocab.contains(BPEBytes.key(merged))) continue;
            return new SelectedPair(entry.pair(), merged);
        }
        return null;
    }

    private static Map<TokenPair, Integer> countPairs(List<TrainingWord> words, int endOfWordId) {
        Map<TokenPair, Integer> counts = new HashMap<>();
        for (TrainingWord word : words) {
            countPairsInWord(word, endOfWordId, counts);
        }
        return counts;
    }

    private static void countPairsInWord(TrainingWord word, int endOfWordId,
                                         Map<TokenPair, Integer> counts) {
        for (int i = 0; i < word.size() - 1; i++) {
            int right = word.get(i + 1);
            if (right == endOfWordId) continue;
            counts.merge(new TokenPair(word.get(i), right), word.frequency(), Integer::sum);
        }
    }

    static TokenPair selectBestPair(Map<TokenPair, Integer> counts) {
        return chooseBestPair(counts, (candidate, count) -> count > 1);
    }

    private static TokenPair chooseBestPair(Map<TokenPair, Integer> counts, PairEligibility eligibility) {
        TokenPair bestPair  = null;
        int       bestCount = 1;

        for (Map.Entry<TokenPair, Integer> entry : counts.entrySet()) {
            TokenPair candidate = entry.getKey();
            int       count     = entry.getValue();

            if (!eligibility.accept(candidate, count)) continue;
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

    private static byte[] mergedBytes(VocabularyState vocab, TokenPair pair) {
        return BPEBytes.concat(
                vocab.idToBytes().get(pair.left()),
                vocab.idToBytes().get(pair.right())
        );
    }

    private static void applyMerge(List<TrainingWord> words, TokenPair pair, int newId,
                                   int endOfWordId, Map<TokenPair, Integer> counts,
                                   PriorityQueue<PairEntry> queue) {
        for (TrainingWord word : words) {
            applyMergeInPlace(word, pair, newId, endOfWordId, counts, queue);
        }
    }

    private static void applyMergeInPlace(TrainingWord word, TokenPair pair, int newId,
                                          int endOfWordId, Map<TokenPair, Integer> counts,
                                          PriorityQueue<PairEntry> queue) {
        int write = 0;
        int read  = 0;
        int size  = word.size();
        while (read < size) {
            if (!matches(word, read, pair)) word.set(write++, word.get(read++));
            else {
                updateNeighborCounts(word, pair, newId, endOfWordId, counts, queue, write, read);
                adjustCount(counts, pair, -word.frequency(), queue);
                word.set(write++, newId);
                read += 2;
            }
        }
        word.truncate(write);
    }

    private static boolean matches(TrainingWord word, int index, TokenPair pair) {
        return index + 1 < word.size()
                && word.get(index) == pair.left()
                && word.get(index + 1) == pair.right();
    }

    private static void updateNeighborCounts(TrainingWord word, TokenPair pair, int newId,
                                             int endOfWordId, Map<TokenPair, Integer> counts,
                                             PriorityQueue<PairEntry> queue, int write, int read) {
        int frequency = word.frequency();
        if (write > 0) updateLeftCounts(word.get(write - 1), pair, newId, frequency, counts, queue);
        int rightIndex = read + 2;
        if (rightIndex >= word.size()) return;
        int right = word.get(rightIndex);
        if (right != endOfWordId) updateRightCounts(right, pair, newId, frequency, counts, queue);
    }

    private static void updateLeftCounts(int left, TokenPair pair, int newId, int frequency,
                                         Map<TokenPair, Integer> counts, PriorityQueue<PairEntry> queue) {
        adjustCount(counts, new TokenPair(left, pair.left()), -frequency, queue);
        adjustCount(counts, new TokenPair(left, newId), frequency, queue);
    }

    private static void updateRightCounts(int right, TokenPair pair, int newId, int frequency,
                                          Map<TokenPair, Integer> counts, PriorityQueue<PairEntry> queue) {
        adjustCount(counts, new TokenPair(pair.right(), right), -frequency, queue);
        adjustCount(counts, new TokenPair(newId, right), frequency, queue);
    }

    private static void adjustCount(Map<TokenPair, Integer> counts, TokenPair pair, int delta,
                                    PriorityQueue<PairEntry> queue) {
        int updated = counts.getOrDefault(pair, 0) + delta;
        if (updated <= 0) {
            counts.remove(pair);
        } else {
            counts.put(pair, updated);
            if (updated > 1) queue.offer(new PairEntry(updated, pair));
        }
    }

    private static String readSample(Path path, int sampleChars) throws IOException {
        StringBuilder sb = new StringBuilder(sampleChars);
        try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            char[] buf = new char[8192];
            int remaining = sampleChars;
            int n;
            while (remaining > 0 && (n = br.read(buf, 0, Math.min(buf.length, remaining))) != -1) {
                sb.append(buf, 0, n);
                remaining -= n;
            }
        }
        return removeDanglingHighSurrogate(sb);
    }

    private static String removeDanglingHighSurrogate(StringBuilder sample) {
        int last = sample.length() - 1;
        if (last >= 0 && Character.isHighSurrogate(sample.charAt(last))) sample.setLength(last);
        return sample.toString();
    }

    private record MergeResult(List<TokenPair> merges, Map<TokenPair, Integer> mergeToNewId) {}

    private record SelectedPair(TokenPair pair, byte[] merged) {}

    private static final class TrainingWord {
        private final int[] tokens;
        private final int frequency;
        private int size;

        private TrainingWord(int[] tokens, int frequency) {
            this.tokens = tokens;
            this.frequency = frequency;
            this.size = tokens.length;
        }

        int get(int index) { return tokens[index]; }

        void set(int index, int token) { tokens[index] = token; }

        int size() { return size; }

        int frequency() { return frequency; }

        void truncate(int newSize) { size = newSize; }
    }

    private record PairEntry(int count, TokenPair pair) implements Comparable<PairEntry> {
        @Override
        public int compareTo(PairEntry other) {
            int cmp = Integer.compare(other.count, this.count);
            return cmp != 0 ? cmp : this.pair.compareTo(other.pair);
        }
    }

    @FunctionalInterface
    private interface PairEligibility {
        boolean accept(TokenPair candidate, int count);
    }
}
