package io.github.kirstenali.deepj.cli;

final class CliText {
    private CliText() {}

    static String safe(String value) {
        StringBuilder result = new StringBuilder();
        value.codePoints().filter(CliText::printable).forEach(result::appendCodePoint);
        return result.toString();
    }

    private static boolean printable(int codePoint) {
        return codePoint == '\n' || codePoint == '\t' || !Character.isISOControl(codePoint);
    }

    static String label(String value) {
        return safe(value).replace('\n', ' ').replace('\t', ' ');
    }
}
