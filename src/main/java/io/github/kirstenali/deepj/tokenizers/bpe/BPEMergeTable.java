package io.github.kirstenali.deepj.tokenizers.bpe;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

final class BPEMergeTable {

    static final int NO_MERGE = -1;

    private final long[] keys;
    private final int[] ranks;
    private final int[] leftByRank;
    private final int[] rightByRank;
    private final int[] resultByRank;
    private final int mask;

    BPEMergeTable(List<TokenPair> merges, Map<TokenPair, Integer> resultIds) {
        int capacity = tableCapacity(merges.size());
        this.keys = new long[capacity];
        this.ranks = new int[capacity];
        this.leftByRank = new int[merges.size()];
        this.rightByRank = new int[merges.size()];
        this.resultByRank = new int[merges.size()];
        this.mask = capacity - 1;
        Arrays.fill(ranks, NO_MERGE);
        indexMerges(merges, resultIds);
    }

    int rank(int left, int right) {
        long key = pairKey(left, right);
        int slot = slot(key);
        while (ranks[slot] != NO_MERGE) {
            if (keys[slot] == key) return ranks[slot];
            slot = (slot + 1) & mask;
        }
        return NO_MERGE;
    }

    int left(int rank) {
        return leftByRank[rank];
    }

    int right(int rank) {
        return rightByRank[rank];
    }

    int result(int rank) {
        return resultByRank[rank];
    }

    private void indexMerges(List<TokenPair> merges, Map<TokenPair, Integer> resultIds) {
        for (int rank = 0; rank < merges.size(); rank++) {
            TokenPair pair = merges.get(rank);
            Integer resultId = resultIds.get(pair);
            if (resultId == null) throw missingResultId(pair);
            index(rank, pair, resultId);
        }
    }

    private void index(int rank, TokenPair pair, int resultId) {
        long key = pairKey(pair.left(), pair.right());
        int slot = slot(key);
        while (ranks[slot] != NO_MERGE) slot = nextDistinctSlot(slot, key);
        keys[slot] = key;
        ranks[slot] = rank;
        leftByRank[rank] = pair.left();
        rightByRank[rank] = pair.right();
        resultByRank[rank] = resultId;
    }

    private int nextDistinctSlot(int slot, long key) {
        if (keys[slot] == key) throw new IllegalArgumentException("Duplicate merge pair");
        return (slot + 1) & mask;
    }

    private int slot(long key) {
        long mixed = key ^ (key >>> 33);
        mixed *= 0xff51afd7ed558ccdl;
        mixed ^= mixed >>> 33;
        return (int) mixed & mask;
    }

    private static long pairKey(int left, int right) {
        return ((long) left << 32) ^ (right & 0xffffffffL);
    }

    private static int tableCapacity(int mergeCount) {
        int required = Math.max(2, mergeCount * 2);
        int capacity = 1;
        while (capacity < required) capacity <<= 1;
        return capacity;
    }

    private static IllegalArgumentException missingResultId(TokenPair pair) {
        return new IllegalArgumentException("Missing result id for merge " + pair);
    }
}
