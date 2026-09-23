package io.github.kirstenali.deepj.tensor.metal;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeSourceStyleTest {

    private static final Path SOURCE = Path.of("native/deepj_metal_jni.mm");

    @Test
    void nativeSourceFollowsReadabilityRules() throws IOException {
        List<String> violations = NativeSourceRules.validate(Files.readAllLines(SOURCE));
        assertTrue(violations.isEmpty(), String.join(System.lineSeparator(), violations));
    }

    @Test
    void nativeSourceHasNoLargeDuplicateBlocks() throws IOException {
        List<String> duplicates = NativeDuplicates.find(Files.readAllLines(SOURCE));
        assertTrue(duplicates.isEmpty(), String.join(System.lineSeparator(), duplicates));
    }

    @Test
    void nativeDuplicateRuleDetectsCopiedBlocks() {
        List<String> block = IntStream.range(0, 16).mapToObj(index -> "line " + index).toList();
        List<String> source = new ArrayList<>(block);
        source.addAll(block);
        assertFalse(NativeDuplicates.find(source).isEmpty());
    }

    private static final class NativeSourceRules {

        private static final int MAX_LINE_LENGTH = 120;

        private static List<String> validate(List<String> lines) {
            List<String> violations = new ArrayList<>();
            for (int index = 0; index < lines.size(); index++) {
                validateLine(lines.get(index), index + 1, violations);
            }
            return violations;
        }

        private static void validateLine(String line, int number, List<String> violations) {
            if (line.length() > MAX_LINE_LENGTH) add(violations, number, "line exceeds 120 characters");
            if (loopNeedsBraces(line)) add(violations, number, "loop body needs braces");
            if (hasMultipleStatements(line)) add(violations, number, "line contains multiple statements");
        }

        private static boolean loopNeedsBraces(String line) {
            String value = line.strip();
            if (!value.startsWith("for (") && !value.startsWith("while (")) return false;
            int closing = value.lastIndexOf(')');
            return closing >= 0 && !value.substring(closing + 1).strip().startsWith("{");
        }

        private static boolean hasMultipleStatements(String line) {
            if (line.strip().startsWith("for (")) return false;
            return line.chars().filter(character -> character == ';').count() > 1;
        }

        private static void add(List<String> violations, int line, String message) {
            violations.add("line " + line + ": " + message);
        }
    }

    private static final class NativeDuplicates {

        private static final int BLOCK_LINES = 16;

        private static List<String> find(List<String> source) {
            List<SourceLine> lines = significantLines(source);
            Map<List<String>, Integer> starts = new HashMap<>();
            List<String> duplicates = new ArrayList<>();
            for (int index = 0; index + BLOCK_LINES <= lines.size(); index++) {
                addDuplicate(lines, index, starts, duplicates);
            }
            return duplicates;
        }

        private static void addDuplicate(List<SourceLine> lines, int index,
                                         Map<List<String>, Integer> starts,
                                         List<String> duplicates) {
            List<String> block = block(lines, index);
            int current = lines.get(index).number();
            Integer previous = starts.putIfAbsent(block, current);
            if (previous != null) duplicates.add("duplicate native block at lines " + previous + " and " + current);
        }

        private static List<SourceLine> significantLines(List<String> source) {
            List<SourceLine> result = new ArrayList<>();
            for (int index = 0; index < source.size(); index++) {
                String line = source.get(index).strip().replaceAll("\\s+", " ");
                if (!line.isEmpty()) result.add(new SourceLine(index + 1, line));
            }
            return result;
        }

        private static List<String> block(List<SourceLine> lines, int start) {
            return lines.subList(start, start + BLOCK_LINES).stream().map(SourceLine::text).toList();
        }

        private record SourceLine(int number, String text) {}
    }
}
