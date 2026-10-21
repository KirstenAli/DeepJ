package io.github.kirstenali.deepj.tensor.metal;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

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

    @TempDir Path directory;

    @ParameterizedTest(name = "{0}")
    @MethodSource("sources")
    void nativeSourceFollowsReadabilityRules(Path file) throws IOException {
        List<String> violations = NativeSourceRules.validate(Files.readAllLines(file));
        assertTrue(violations.isEmpty(), file + ": " + String.join("\n", violations));
    }

    private static List<Path> sources() throws IOException {
        return NativeSourceFiles.all();
    }

    @Test
    void nativeSourceHasNoLargeDuplicateBlocks() throws IOException {
        List<String> duplicates = NativeDuplicates.findSources(NativeSourceFiles.all());
        assertTrue(duplicates.isEmpty(), String.join(System.lineSeparator(), duplicates));
    }

    @Test
    void nativeDuplicateRuleDetectsCopiedBlocks() {
        List<String> block = IntStream.range(0, 16).mapToObj(index -> "line " + index).toList();
        List<String> source = new ArrayList<>(block);
        source.addAll(block);
        assertFalse(NativeDuplicates.find(source).isEmpty());
    }

    @Test
    void nativeDuplicateRuleDetectsCopiesAcrossTestHeaders() throws IOException {
        List<String> block = IntStream.range(0, 16).mapToObj(index -> "line " + index).toList();
        Path first = Files.write(directory.resolve("first.h"), block);
        Path second = Files.write(directory.resolve("second.h"), block);
        assertFalse(NativeDuplicates.findSources(List.of(first, second)).isEmpty());
    }

    @Test
    void nativeStyleRulesRejectLongLinesUnbracedLoopsAndMultipleStatements() {
        assertFalse(NativeSourceRules.validate(List.of("//" + "x".repeat(119))).isEmpty());
        assertFalse(NativeSourceRules.validate(List.of("for (int i = 0; i < 2; i++) work();")).isEmpty());
        assertFalse(NativeSourceRules.validate(List.of("first(); second();")).isEmpty());
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
            List<String> duplicates = new ArrayList<>();
            addSource(source, "fixture", new HashMap<>(), duplicates);
            return duplicates;
        }

        private static List<String> findSources(List<Path> files) throws IOException {
            Map<List<String>, String> starts = new HashMap<>();
            List<String> duplicates = new ArrayList<>();
            for (Path file : files) {
                addSource(Files.readAllLines(file), file.toString(), starts, duplicates);
            }
            return duplicates;
        }

        private static void addSource(List<String> source, String file,
                                      Map<List<String>, String> starts, List<String> duplicates) {
            List<SourceLine> lines = significantLines(source);
            for (int index = 0; index + BLOCK_LINES <= lines.size(); index++) {
                addDuplicate(lines, index, file, starts, duplicates);
            }
        }

        private static void addDuplicate(List<SourceLine> lines, int index, String file,
                                         Map<List<String>, String> starts,
                                         List<String> duplicates) {
            List<String> block = block(lines, index);
            String current = file + ":" + lines.get(index).number();
            String previous = starts.putIfAbsent(block, current);
            if (previous != null) duplicates.add("duplicate native block at " + previous + " and " + current);
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
