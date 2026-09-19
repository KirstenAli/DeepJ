package io.github.kirstenali.deepj.data;

final class StatefulRandom {

    private static final long INCREMENT = 0x9E3779B97F4A7C15L;
    private long state;

    StatefulRandom(long seed) {
        state = seed;
    }

    long state() {
        return state;
    }

    void restore(long savedState) {
        state = savedState;
    }

    long nextLong(long bound) {
        if (bound <= 0) throw new IllegalArgumentException("bound must be positive");
        long random = nextLong() >>> 1;
        long value = random % bound;
        while (random - value + bound - 1 < 0) {
            random = nextLong() >>> 1;
            value = random % bound;
        }
        return value;
    }

    private long nextLong() {
        state += INCREMENT;
        long value = state;
        value = (value ^ value >>> 30) * 0xBF58476D1CE4E5B9L;
        value = (value ^ value >>> 27) * 0x94D049BB133111EBL;
        return value ^ value >>> 31;
    }
}
