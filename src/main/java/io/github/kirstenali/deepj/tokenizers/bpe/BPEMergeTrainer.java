package io.github.kirstenali.deepj.tokenizers.bpe;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

final class BPEMergeTrainer {

    private BPEMergeTrainer() {}

    static Result train(String text, VocabularyState vocab, int targetVocabSize) {
        List<TrainingWord> words = buildInitialWords(text, vocab.endOfWordId());
        List<TokenPair> merges = new ArrayList<>();
        Map<TokenPair, Integer> mergeIds = new HashMap<>();
        Map<TokenPair, Integer> counts = countPairs(words, vocab.endOfWordId());
        PriorityQueue<PairEntry> queue = buildQueue(counts);
        trainMerges(words, vocab, targetVocabSize, merges, mergeIds, counts, queue);
        return new Result(merges, mergeIds);
    }

    private static void trainMerges(List<TrainingWord> words, VocabularyState vocab, int targetSize,
                                    List<TokenPair> merges, Map<TokenPair, Integer> mergeIds,
                                    Map<TokenPair, Integer> counts, PriorityQueue<PairEntry> queue) {
        while (vocab.size() < targetSize) {
            SelectedPair best = pollBestValidPair(queue, counts, vocab);
            if (best == null) return;
            mergeBestPair(words, vocab, merges, mergeIds, counts, queue, best);
        }
    }

    private static void mergeBestPair(List<TrainingWord> words, VocabularyState vocab,
                                      List<TokenPair> merges, Map<TokenPair, Integer> mergeIds,
                                      Map<TokenPair, Integer> counts, PriorityQueue<PairEntry> queue,
                                      SelectedPair best) {
        int newId = vocab.add(best.merged(), BPEBytes.key(best.merged()));
        merges.add(best.pair());
        mergeIds.put(best.pair(), newId);
        applyMerge(words, best.pair(), newId, vocab.endOfWordId(), counts, queue);
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
        while (end < text.length() && Character.isWhitespace(text.charAt(end)) == whitespace) {
            end++;
        }
        return end;
    }

    private static TrainingWord trainingWord(String piece, int endOfWordId, int frequency) {
        return new TrainingWord(BPEBytes.toTokenArray(piece, endOfWordId), frequency);
    }

    private static PriorityQueue<PairEntry> buildQueue(Map<TokenPair, Integer> counts) {
        PriorityQueue<PairEntry> queue = new PriorityQueue<>(Math.max(1, counts.size()));
        for (Map.Entry<TokenPair, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > 1) queue.offer(new PairEntry(entry.getValue(), entry.getKey()));
        }
        return queue;
    }

    private static SelectedPair pollBestValidPair(PriorityQueue<PairEntry> queue,
                                                  Map<TokenPair, Integer> counts,
                                                  VocabularyState vocab) {
        while (!queue.isEmpty()) {
            PairEntry entry = queue.poll();
            Integer current = counts.get(entry.pair());
            if (current == null || current != entry.count() || current <= 1) continue;
            byte[] merged = mergedBytes(vocab, entry.pair());
            if (!vocab.contains(BPEBytes.key(merged))) return new SelectedPair(entry.pair(), merged);
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
        for (int index = 0; index < word.size() - 1; index++) {
            int right = word.get(index + 1);
            if (right == endOfWordId) continue;
            counts.merge(new TokenPair(word.get(index), right), word.frequency(), Integer::sum);
        }
    }

    static TokenPair selectBestPair(Map<TokenPair, Integer> counts) {
        return chooseBestPair(counts, (candidate, count) -> count > 1);
    }

    private static TokenPair chooseBestPair(Map<TokenPair, Integer> counts, PairEligibility eligibility) {
        TokenPair bestPair = null;
        int bestCount = 1;
        for (Map.Entry<TokenPair, Integer> entry : counts.entrySet()) {
            if (!eligibility.accept(entry.getKey(), entry.getValue())) continue;
            if (!isBetterCandidate(entry.getKey(), entry.getValue(), bestPair, bestCount)) continue;
            bestPair = entry.getKey();
            bestCount = entry.getValue();
        }
        return bestPair;
    }

    private static boolean isBetterCandidate(TokenPair candidate, int count,
                                             TokenPair bestPair, int bestCount) {
        if (count > bestCount) return true;
        return count == bestCount && (bestPair == null || candidate.compareTo(bestPair) < 0);
    }

    private static byte[] mergedBytes(VocabularyState vocab, TokenPair pair) {
        return BPEBytes.concat(vocab.idToBytes().get(pair.left()), vocab.idToBytes().get(pair.right()));
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
        int read = 0;
        int size = word.size();
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
        if (updated <= 0) counts.remove(pair);
        else {
            counts.put(pair, updated);
            if (updated > 1) queue.offer(new PairEntry(updated, pair));
        }
    }

    record Result(List<TokenPair> merges, Map<TokenPair, Integer> mergeToNewId) {}

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
            int comparison = Integer.compare(other.count, count);
            return comparison != 0 ? comparison : pair.compareTo(other.pair);
        }
    }

    @FunctionalInterface
    private interface PairEligibility {
        boolean accept(TokenPair candidate, int count);
    }
}
