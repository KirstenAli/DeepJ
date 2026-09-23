package io.github.kirstenali.deepj.tokenizers.bpe;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class BPETokenizerTest {

    @Test
    void encodeDecode_roundTripsAscii() {
        String trainingText = "low lower lowest low lower lowest";
        String input = "low lowest";

        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer(trainingText, 280);

        int[] ids = tokenizer.encode(input);
        String decoded = tokenizer.decode(ids);

        assertEquals(input, decoded);
    }

    @Test
    void encodeDecode_roundTripsWhitespaceExactly() {
        String trainingText = "hello   world\nhello   world\n";
        String input = "hello   world\n";

        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer(trainingText, 280);

        int[] ids = tokenizer.encode(input);
        String decoded = tokenizer.decode(ids);

        assertEquals(input, decoded);
    }

    @Test
    void encodeDecode_roundTripsUnicode() {
        String trainingText = "héllo héllo café café 🚀🚀";
        String input = "héllo café 🚀";

        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer(trainingText, 290);

        int[] ids = tokenizer.encode(input);
        String decoded = tokenizer.decode(ids);

        assertEquals(input, decoded);
    }

    @Test
    void encodingAfterTrainingUsuallyUsesFewerTokensForFrequentPatterns() {
        String trainingText = "banana banana banana banana banana";
        String input = "banana";

        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer(trainingText, 270);

        int[] ids = tokenizer.encode(input);

        assertTrue(ids.length < input.getBytes().length,
                "Expected trained tokenizer to compress frequent pattern into fewer tokens");
    }

    @Test
    void decode_rejectsInvalidTokenIds() {
        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer("hello hello hello", 270);

        assertThrows(IllegalArgumentException.class, () -> tokenizer.decode(new int[]{999999}));
    }

    @Test
    void encode_emptyStringProducesNoTokens() {
        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer("hello world", 270);

        int[] ids = tokenizer.encode("");

        assertArrayEquals(new int[0], ids);
    }

    @Test
    void encodeDecode_handlesConfiguredSpecialTokensAtomically() {
        BPETrainer trainer = new BPETrainer();
        BPETokenizer tokenizer = trainer.trainTokenizer(
                "hello world hello world",
                280,
                List.of("<BOS>", "<EOS>", "<PAD>")
        );

        int[] ids = tokenizer.encode("<BOS> hello <EOS>");
        assertEquals(tokenizer.model().specialTokenToId().get("<BOS>"), ids[0]);
        assertEquals(tokenizer.model().specialTokenToId().get("<EOS>"), ids[ids.length - 1]);
        assertEquals("<BOS> hello <EOS>", tokenizer.decode(ids));
    }

    @Test
    void recognizesConfiguredEndOfSequenceTokens() {
        BPETokenizer tokenizer = new BPETrainer().trainTokenizer(
                "hello world hello world", 280,
                List.of("<BOS>", "<EOS>", "<PAD>", "<|endoftext|>"));

        assertTrue(tokenizer.isEndOfSequence(specialId(tokenizer, "<EOS>")));
        assertTrue(tokenizer.isEndOfSequence(specialId(tokenizer, "<|endoftext|>")));
        assertFalse(tokenizer.isEndOfSequence(specialId(tokenizer, "<PAD>")));
    }

    @Test
    void rankedEncoderMatchesOrderedMergeReference() {
        BPETokenizer tokenizer = new BPETrainer().trainTokenizer(
                "banana bandana banana café café 🚀", 290);

        for (String text : List.of("banana", "bandana café", "🚀 banana", "unknown")) {
            assertArrayEquals(referenceEncode(tokenizer.model(), text), tokenizer.encode(text));
        }
    }

    private static int[] referenceEncode(BPEModel model, String text) {
        List<Integer> result = new ArrayList<>();
        for (String piece : BPEBytes.splitPreserveWhitespace(text)) {
            result.addAll(referencePiece(model, piece));
        }
        return result.stream().mapToInt(Integer::intValue).toArray();
    }

    private static int specialId(BPETokenizer tokenizer, String token) {
        return tokenizer.model().specialTokenToId().get(token);
    }

    private static List<Integer> referencePiece(BPEModel model, String piece) {
        int[] initial = BPEBytes.toTokenArray(piece, model.endOfWordId());
        List<Integer> tokens = new ArrayList<>(initial.length);
        for (int id : initial) {
            tokens.add(id);
        }
        for (TokenPair pair : model.merges()) {
            tokens = merge(tokens, pair, model.mergeToNewId().get(pair));
        }
        if (!tokens.isEmpty() && tokens.get(tokens.size() - 1) == model.endOfWordId()) tokens.remove(tokens.size() - 1);
        return tokens;
    }

    private static List<Integer> merge(List<Integer> tokens, TokenPair pair, int resultId) {
        List<Integer> merged = new ArrayList<>(tokens.size());
        for (int i = 0; i < tokens.size();) {
            if (i + 1 < tokens.size() && tokens.get(i) == pair.left() && tokens.get(i + 1) == pair.right()) {
                merged.add(resultId);
                i += 2;
            } else {
                merged.add(tokens.get(i++));
            }
        }
        return merged;
    }
}
