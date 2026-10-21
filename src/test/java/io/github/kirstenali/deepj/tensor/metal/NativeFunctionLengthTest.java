package io.github.kirstenali.deepj.tensor.metal;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

class NativeFunctionLengthTest {

    private static final Path SOURCE = Path.of("native/deepj_metal_jni.mm");
    private static final int MAX_LINES = 20;

    @ParameterizedTest(name = "{0}")
    @MethodSource("sources")
    void nativeFunctionsStaySmall(Path file) throws IOException {
        ScanResult result = scan(Files.readAllLines(file));
        assertTrue(result.violations().isEmpty(), file + ": " + String.join("\n", result.violations()));
    }

    private static List<Path> sources() throws IOException {
        return NativeSourceFiles.all();
    }

    @Test
    void scannerRecognisesProductionAndTestHelperFunctions() throws IOException {
        assertTrue(scan(Files.readAllLines(SOURCE)).functionCount() > 100);
        assertTrue(scan(Files.readAllLines(Path.of("src/test/native/coverage-hooks.h"))).functionCount() >= 4);
    }

    @ParameterizedTest
    @ValueSource(ints = {20, 21})
    void lengthBoundaryIsEnforcedForHeaderStyleHelpers(int count) {
        String function = "static void helper() {\n" + "    work();\n".repeat(count - 2) + "}";
        ScanResult result = scan(function.lines().toList());
        assertEquals(1, result.functionCount());
        assertEquals(count <= MAX_LINES, result.violations().isEmpty());
    }

    private static ScanResult scan(List<String> lines) {
        FunctionScanner scanner = new FunctionScanner();
        for (int index = 0; index < lines.size(); index++) {
            scanner.accept(lines.get(index), index + 1);
        }
        return scanner.result();
    }

    private record ScanResult(int functionCount, List<String> violations) {}

    private static final class FunctionScanner {

        private static final Pattern NAME = Pattern.compile("([A-Za-z_][\\w:]*)\\s*\\([^;]*\\)\\s*\\{");
        private final List<String> violations = new ArrayList<>();
        private final StringBuilder signature = new StringBuilder();
        private int start;
        private int lines;
        private int depth;
        private int functions;
        private String name;

        void accept(String line, int number) {
            if (isSourceDelimiter(line)) return;
            if (name != null) acceptBody(line);
            else if (!signature.isEmpty()) acceptSignature(line);
            else if (isFunctionStart(line)) beginSignature(line, number);
        }

        ScanResult result() {
            return new ScanResult(functions, List.copyOf(violations));
        }

        private boolean isSourceDelimiter(String line) {
            return line.contains("@R\"(") || line.strip().equals(")\";");
        }

        private boolean isFunctionStart(String line) {
            String value = line.strip();
            return value.startsWith("kernel ") || value.startsWith("inline ")
                    || value.startsWith("static ") || value.startsWith("extern \"C\" JNIEXPORT");
        }

        private void beginSignature(String line, int number) {
            start = number;
            appendLine(line);
            if (line.contains(";") && !line.contains("{")) reset();
            else readOpeningBrace(line);
        }

        private void acceptSignature(String line) {
            appendLine(line);
            if (line.contains("{")) readOpeningBrace(line);
            else if (line.contains(";")) reset();
        }

        private void readOpeningBrace(String line) {
            if (!line.contains("{")) return;
            Matcher matcher = NAME.matcher(signature);
            if (!matcher.find()) {
                reset();
                return;
            }
            name = matcher.group(1);
            depth = braceDelta(signature.toString());
            if (depth == 0) finish();
        }

        private void acceptBody(String line) {
            appendLine(line);
            depth += braceDelta(line);
            if (depth == 0) finish();
        }

        private void appendLine(String line) {
            signature.append(line).append('\n');
            if (!line.isBlank()) lines++;
        }

        private void finish() {
            functions++;
            if (lines > MAX_LINES) {
                violations.add(name + " starts at line " + start + " and has " + lines + " lines");
            }
            reset();
        }

        private void reset() {
            signature.setLength(0);
            start = 0;
            lines = 0;
            depth = 0;
            name = null;
        }

        private static int braceDelta(String value) {
            return Math.toIntExact(value.chars().filter(character -> character == '{').count()
                    - value.chars().filter(character -> character == '}').count());
        }
    }
}
