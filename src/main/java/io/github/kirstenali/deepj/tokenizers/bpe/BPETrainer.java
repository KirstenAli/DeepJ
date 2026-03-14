package io.github.kirstenali.deepj.tokenizers.bpe;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class BPETrainer {

    private static final byte[] END_OF_WORD = new byte[0];
    private static final String EOW_KEY = "<EOW_INTERNAL>";

    public BPEModel train(String text, int targetVocabSize) {
        validateTargetVocabSize(targetVocabSize);

        VocabularyState vocab = createBaseVocabulary();
        List<List<Integer>> words = buildInitialWords(text, vocab.endOfWordId());

        List<TokenPair> merges = new ArrayList<>();
        Map<TokenPair, Integer> mergeToNewId = new HashMap<>();

        while (vocab.size() < targetVocabSize) {
            TokenPair bestPair = findBestPair(words, vocab.endOfWordId());
            if (bestPair == null) {
                break;
            }

            Integer newId = tryAddMergedToken(vocab, bestPair);
            if (newId == null) {
                break;
            }

            merges.add(bestPair);
            mergeToNewId.put(bestPair, newId);
            applyMerge(words, bestPair, newId);
        }

        return new BPEModel(
                vocab.idToBytes(),
                vocab.bytesToId(),
                merges,
                mergeToNewId,
                vocab.endOfWordId()
        );
    }

    public BPEModel trainFromFile(Path path, int targetVocabSize) throws IOException {
        return train(Files.readString(path), targetVocabSize);
    }

    public BPETokenizer trainTokenizer(String text, int targetVocabSize) {
        return new BPETokenizer(train(text, targetVocabSize));
    }

    public BPETokenizer trainTokenizerFromFile(Path path, int targetVocabSize) throws IOException {
        return new BPETokenizer(trainFromFile(path, targetVocabSize));
    }

    private void validateTargetVocabSize(int targetVocabSize) {
        if (targetVocabSize <= 257) {
            throw new IllegalArgumentException("targetVocabSize must be > 257");
        }
    }

    private VocabularyState createBaseVocabulary() {
        List<byte[]> idToBytes = new ArrayList<>();
        Map<String, Integer> bytesToId = new HashMap<>();

        for (int i = 0; i < 256; i++) {
            byte[] token = new byte[]{(byte) i};
            idToBytes.add(token);
            bytesToId.put(BPEBytes.key(token), i);
        }

        int endOfWordId = idToBytes.size();
        idToBytes.add(END_OF_WORD);
        bytesToId.put(EOW_KEY, endOfWordId);

        return new VocabularyState(idToBytes, bytesToId, endOfWordId);
    }

    private List<List<Integer>> buildInitialWords(String text, int endOfWordId) {
        List<List<Integer>> words = new ArrayList<>();

        for (String piece : BPEBytes.splitPreserveWhitespace(text)) {
            words.add(toTokenIds(piece, endOfWordId));
        }

        return words;
    }

    static List<Integer> toTokenIds(String piece, int endOfWordId) {
        byte[] bytes = BPEBytes.utf8(piece);
        List<Integer> word = new ArrayList<>(bytes.length + 1);

        for (byte b : bytes) {
            word.add(b & 0xFF);
        }

        word.add(endOfWordId);
        return word;
    }

    private TokenPair findBestPair(List<List<Integer>> words, int endOfWordId) {
        return selectBestPair(countPairs(words, endOfWordId));
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

            if (count > bestCount) {
                bestCount = count;
                bestPair = candidate;
            } else if (count == bestCount && bestPair != null && candidate.compareTo(bestPair) < 0) {
                bestPair = candidate;
            }
        }

        return bestPair;
    }

    private Integer tryAddMergedToken(VocabularyState vocab, TokenPair pair) {
        byte[] mergedBytes = BPEBytes.concat(
                vocab.idToBytes().get(pair.left()),
                vocab.idToBytes().get(pair.right())
        );

        String mergedKey = BPEBytes.key(mergedBytes);
        if (vocab.contains(mergedKey)) {
            return null;
        }

        return vocab.add(mergedBytes, mergedKey);
    }

    private void applyMerge(List<List<Integer>> words, TokenPair pair, int newId) {
        for (List<Integer> word : words) {
            replacePairInWord(word, pair, newId);
        }
    }

    private void replacePairInWord(List<Integer> word, TokenPair pair, int newId) {
        List<Integer> merged = new ArrayList<>(word.size());
        int i = 0;

        while (i < word.size()) {
            if (i < word.size() - 1
                    && word.get(i) == pair.left()
                    && word.get(i + 1) == pair.right()) {
                merged.add(newId);
                i += 2;
            } else {
                merged.add(word.get(i));
                i++;
            }
        }

        word.clear();
        word.addAll(merged);
    }
}