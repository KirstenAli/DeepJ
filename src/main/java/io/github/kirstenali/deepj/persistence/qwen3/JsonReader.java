package io.github.kirstenali.deepj.persistence.qwen3;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class JsonReader {
    private final String text;
    private int cursor, depth;

    JsonReader(String text) {
        if (text == null || text.length() > 64 * 1024 * 1024) throw new IllegalArgumentException("Invalid JSON size");
        this.text = text;
    }

    Object read() {
        Object value = value();
        whitespace();
        if (cursor != text.length()) throw error("Trailing JSON content");
        return value;
    }

    private Object value() {
        whitespace();
        if (++depth > 64 || cursor == text.length()) throw error("JSON is too deep or incomplete");
        try { return token(); }
        finally { depth--; }
    }

    private Object token() {
        return switch (text.charAt(cursor)) {
            case '{' -> object();
            case '[' -> array();
            case '"' -> string();
            case 't' -> literal("true", Boolean.TRUE);
            case 'f' -> literal("false", Boolean.FALSE);
            case 'n' -> literal("null", null);
            default -> number();
        };
    }

    private Map<String, Object> object() {
        cursor++;
        Map<String, Object> result = new LinkedHashMap<>();
        if (consume('}')) return result;
        do {
            whitespace();
            String name = string();
            if (result.containsKey(name)) throw error("Duplicate JSON key");
            require(':');
            result.put(name, value());
        } while (consume(','));
        require('}');
        return result;
    }

    private List<Object> array() {
        cursor++;
        List<Object> result = new ArrayList<>();
        if (consume(']')) return result;
        do {
            result.add(value());
        } while (consume(','));
        require(']');
        return result;
    }

    private String string() {
        require('"');
        StringBuilder result = new StringBuilder();
        while (cursor < text.length()) {
            char character = text.charAt(cursor++);
            if (character == '"') return result.toString();
            if (character < 32) throw error("Control character in JSON string");
            result.append(character == '\\' ? escape() : character);
        }
        throw error("Unterminated JSON string");
    }

    private char escape() {
        if (cursor == text.length()) throw error("Incomplete JSON escape");
        char code = text.charAt(cursor++);
        int index = "\"\\/bfnrt".indexOf(code);
        if (index >= 0) return "\"\\/\b\f\n\r\t".charAt(index);
        if (code == 'u') return unicode();
        throw error("Unknown JSON escape");
    }

    private char unicode() {
        if (cursor + 4 > text.length()) throw error("Incomplete Unicode escape");
        int value = 0;
        for (int index = 0; index < 4; index++) {
            int digit = Character.digit(text.charAt(cursor++), 16);
            if (digit < 0) throw error("Invalid Unicode escape");
            value = value * 16 + digit;
        }
        return (char) value;
    }

    private Object number() {
        String token = numberToken();
        try {
            if (!token.contains(".") && !token.contains("e") && !token.contains("E")) return Long.valueOf(token);
            double result = Double.parseDouble(token);
            if (!Double.isFinite(result)) throw error("Non-finite JSON number");
            return result;
        } catch (NumberFormatException failure) { throw error("Invalid JSON number"); }
    }

    private String numberToken() {
        int start = cursor;
        match('-');
        if (!match('0')) digits();
        if (match('.')) digits();
        if (match('e') || match('E')) exponent();
        return text.substring(start, cursor);
    }

    private void exponent() {
        if (!match('+')) match('-');
        digits();
    }

    private void digits() {
        int start = cursor;
        while (cursor < text.length() && text.charAt(cursor) >= '0' && text.charAt(cursor) <= '9') { cursor++; }
        if (cursor == start) throw error("Expected JSON digits");
    }

    private Object literal(String expected, Object value) {
        if (!text.startsWith(expected, cursor)) throw error("Invalid JSON literal");
        cursor += expected.length();
        return value;
    }

    private boolean consume(char expected) {
        whitespace();
        return match(expected);
    }

    private boolean match(char expected) {
        if (cursor == text.length() || text.charAt(cursor) != expected) return false;
        cursor++;
        return true;
    }

    private void require(char expected) {
        if (!consume(expected)) throw error("Expected '" + expected + "'");
    }

    private void whitespace() {
        while (cursor < text.length() && " \t\r\n".indexOf(text.charAt(cursor)) >= 0) { cursor++; }
    }

    private IllegalArgumentException error(String message) {
        return new IllegalArgumentException(message + " at character " + cursor);
    }
}
