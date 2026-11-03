package io.github.kirstenali.deepj.tokenizers.qwen3;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class BytePairMerges {
    private final Map<Pair, Integer> ranks = new HashMap<>();
    record Pair(String left, String right) {}

    BytePairMerges(List<?> merges, Map<String, Integer> vocabulary) {
        for (int rank = 0; rank < merges.size(); rank++) {
            List<?> parts = parts(merges.get(rank));
            if (parts.size() != 2) throw new IllegalArgumentException("BPE merge must have two symbols");
            Pair pair = new Pair(Qwen3Json.string(parts.get(0)), Qwen3Json.string(parts.get(1)));
            if (!vocabulary.containsKey(pair.left() + pair.right()) || ranks.put(pair, rank) != null) {
                throw new IllegalArgumentException("Invalid or repeated BPE merge");
            }
        }
    }

    private static List<?> parts(Object value) {
        return value instanceof String line ? List.of(line.split(" ", -1)) : Qwen3Json.array(value);
    }

    List<String> apply(String text) {
        List<String> pieces = new ArrayList<>();
        text.chars().forEach(symbol -> pieces.add(String.valueOf((char) symbol)));
        int best;
        while ((best = bestPair(pieces)) >= 0) {
            pieces.set(best, pieces.get(best) + pieces.remove(best + 1));
        }
        return pieces;
    }

    private int bestPair(List<String> pieces) {
        int position = -1, best = Integer.MAX_VALUE;
        for (int index = 0; index + 1 < pieces.size(); index++) {
            int rank = ranks.getOrDefault(new Pair(pieces.get(index), pieces.get(index + 1)), Integer.MAX_VALUE);
            if (rank < best) {
                best = rank;
                position = index;
            }
        }
        return position;
    }
}
