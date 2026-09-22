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
        BPEMergeTrainer.Result result = BPEMergeTrainer.train(text, vocab, targetVocabSize);

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

    public BPETokenizer trainTokenizerFromFile(Path path, int targetVocabSize,
                                               List<String> specialTokens) throws IOException {
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

    static TokenPair selectBestPair(Map<TokenPair, Integer> counts) {
        return BPEMergeTrainer.selectBestPair(counts);
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

}
