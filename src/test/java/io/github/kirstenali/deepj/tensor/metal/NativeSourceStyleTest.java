package io.github.kirstenali.deepj.tensor.metal;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeSourceStyleTest {

    private static final Path SOURCE = Path.of("native/deepj_metal_jni.mm");

    @Test
    void nativeSourceFollowsReadabilityRules() throws IOException {
        List<String> violations = NativeSourceRules.validate(Files.readAllLines(SOURCE));
        assertTrue(violations.isEmpty(), String.join(System.lineSeparator(), violations));
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
}
