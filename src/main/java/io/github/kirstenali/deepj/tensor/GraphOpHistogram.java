package io.github.kirstenali.deepj.tensor;

import java.lang.reflect.Field;
import java.util.Map;
import java.util.TreeMap;

/** Opt-in opcode counter for the executed GPU command stream (-Ddeepj.graph.histogram=true). */
public final class GraphOpHistogram {
    private static final boolean ENABLED = Boolean.getBoolean("deepj.graph.histogram");
    private static final Map<Integer, Long> COUNTS = new TreeMap<>();

    private GraphOpHistogram() {}

    static void record(int[] commands, int length) {
        if (!ENABLED) return;
        int position = 0;
        while (position < length) {
            int opcode = commands[position];
            count(opcode);
            position += GraphOpMetadata.OP_METADATA[opcode].stride();
        }
    }

    private static void count(int opcode) {
        synchronized (COUNTS) {
            COUNTS.merge(opcode, 1L, Long::sum);
        }
    }

    public static Map<Integer, Long> snapshot() {
        synchronized (COUNTS) {
            return new TreeMap<>(COUNTS);
        }
    }

    public static void reset() {
        synchronized (COUNTS) {
            COUNTS.clear();
        }
    }

    public static Map<Integer, String> opcodeNames() {
        Map<Integer, String> names = new TreeMap<>();
        for (Field field : GraphOpMetadata.class.getDeclaredFields()) {
            if (field.getName().startsWith("OP_") && field.getType() == int.class) {
                names.put(opcode(field), field.getName());
            }
        }
        return names;
    }

    private static int opcode(Field field) {
        try {
            return field.getInt(null);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }
}

