package io.github.kirstenali.deepj.tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Stable index-only metadata gives each GPU scatter thread one independent destination. */
final class IndexGroups {
    private IndexGroups() {}

    static long[] describe(int[] indices) {
        Map<Integer, List<Integer>> groups = new LinkedHashMap<>();
        for (int position = 0; position < indices.length; position++) {
            groups.computeIfAbsent(indices[position], ignored -> new ArrayList<>()).add(position);
        }
        int count = indices.length, distinct = groups.size();
        long[] words = new long[Math.addExact(3, Math.multiplyExact(2, Math.addExact(count, distinct)))];
        words[0] = count;
        words[1] = distinct;
        for (int position = 0; position < count; position++) { words[2 + position] = indices[position]; }
        appendGroups(words, count, groups);
        return words;
    }

    private static void appendGroups(long[] words, int count, Map<Integer, List<Integer>> groups) {
        int group = 0, position = 0, distinct = groups.size();
        int targets = 2 + count, offsets = targets + distinct, positions = offsets + distinct + 1;
        for (var entry : groups.entrySet()) {
            words[targets + group] = entry.getKey();
            words[offsets + group++] = position;
            for (int source : entry.getValue()) { words[positions + position++] = source; }
        }
        words[offsets + distinct] = count;
    }
}
