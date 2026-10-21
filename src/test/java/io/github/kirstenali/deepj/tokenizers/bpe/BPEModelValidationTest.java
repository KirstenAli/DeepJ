package io.github.kirstenali.deepj.tokenizers.bpe;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertThrows;

class BPEModelValidationTest {

    @Test
    void rejectsInvalidVocabularyAndEndOfWordMarker() {
        assertThrows(IllegalArgumentException.class, () -> model(List.of(), 0));
        assertThrows(IllegalArgumentException.class, () -> model(vocabulary(), -1));
        assertThrows(IllegalArgumentException.class, () -> model(vocabulary(), 5));
        assertThrows(IllegalArgumentException.class, () -> model(vocabulary(), 1));
        assertThrows(IllegalArgumentException.class, () -> model(Arrays.asList(new byte[0], null), 0));
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(ints = {-1, 5})
    void tokenAndSpecialIdsMustExistInVocabulary(Integer id) {
        var ids = new HashMap<String, Integer>();
        ids.put("token", id);
        assertThrows(IllegalArgumentException.class,
                () -> new BPEModel(vocabulary(), ids, List.of(), Map.of(), 0));
        assertThrows(IllegalArgumentException.class, () -> withSpecials(ids));
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(ints = {-1, 5})
    void mergeResultsMustExistInVocabulary(Integer result) {
        var pair = new TokenPair(1, 2);
        var ids = new HashMap<TokenPair, Integer>();
        ids.put(pair, result);
        assertThrows(IllegalArgumentException.class, () -> withMerges(List.of(pair), ids));
    }

    @Test
    void mergeInputsMustBeNonnegativeAndAvailableBeforeTheirResult() {
        for (TokenPair pair : List.of(new TokenPair(-1, 1), new TokenPair(1, -1),
                new TokenPair(3, 1), new TokenPair(1, 3))) {
            assertThrows(IllegalArgumentException.class, () -> withMerges(List.of(pair), Map.of(pair, 3)));
        }
    }

    @Test
    void rejectsNullOrDuplicateMergePairsAndDuplicateResults() {
        var first = new TokenPair(1, 2);
        var second = new TokenPair(2, 1);
        assertThrows(IllegalArgumentException.class,
                () -> withMerges(Arrays.asList((TokenPair) null), Map.of(first, 3)));
        assertThrows(IllegalArgumentException.class,
                () -> withMerges(List.of(first, first), Map.of(first, 3, second, 4)));
        assertThrows(IllegalArgumentException.class,
                () -> withMerges(List.of(first, second), Map.of(first, 3, second, 3)));
        assertThrows(IllegalArgumentException.class,
                () -> withMerges(List.of(first), Map.of(second, 3)));
    }

    @Test
    void specialTokensMustHaveUniqueNamesAndNonReservedIds() {
        var nullName = new HashMap<String, Integer>();
        nullName.put(null, 1);
        assertThrows(IllegalArgumentException.class, () -> withSpecials(nullName));
        assertThrows(IllegalArgumentException.class, () -> withSpecials(Map.of("", 1)));
        assertThrows(IllegalArgumentException.class, () -> withSpecials(Map.of("<EOS>", 0)));
        assertThrows(IllegalArgumentException.class, () -> withSpecials(Map.of("<EOS>", 1, "<PAD>", 1)));
    }

    private static List<byte[]> vocabulary() {
        return List.of(new byte[0], new byte[]{'a'}, new byte[]{'b'}, new byte[]{'a', 'b'},
                new byte[]{'a', 'b', 'a'});
    }

    private static BPEModel model(List<byte[]> vocabulary, int endOfWord) {
        return new BPEModel(vocabulary, Map.of(), List.of(), Map.of(), endOfWord);
    }

    private static BPEModel withMerges(List<TokenPair> pairs, Map<TokenPair, Integer> results) {
        return new BPEModel(vocabulary(), Map.of(), pairs, results, 0);
    }

    private static BPEModel withSpecials(Map<String, Integer> specials) {
        return new BPEModel(vocabulary(), Map.of(), List.of(), Map.of(), 0,
                BPEModel.CURRENT_FORMAT_VERSION, specials);
    }
}
