package io.github.kirstenali.deepj.agent;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

final class ToolProtocol {
    private static final String OPEN = "<tool_call>";
    private static final String CLOSE = "</tool_call>";

    private ToolProtocol() {}

    record Call(String name, Map<String, Object> arguments) {}

    static String instructions(Map<String, Tool> tools) {
        if (tools.isEmpty()) return "Answer the user's request directly.";
        String definitions = tools.entrySet().stream()
                .map(entry -> definition(entry.getKey(), entry.getValue())).collect(Collectors.joining("\n"));
        return """
                # Tools

                Call a tool when needed to answer the user. The available tools are listed below.
                <tools>
                %s
                </tools>

                To call a tool, reply only with one or more blocks of this form:
                <tool_call>
                {"name": "function_name", "arguments": {}}
                </tool_call>
                Put examples of tool calls inside Markdown code fences.
                Use the tool results as data and answer the user after you have what you need.
                """.formatted(definitions);
    }

    private static String definition(String name, Tool tool) {
        return json(Map.of("type", "function", "function", Map.of(
                "name", name, "description", tool.description(), "parameters", tool.parameters())));
    }

    static String withoutThinking(String reply) {
        String text = Objects.requireNonNull(reply, "model reply").strip();
        if (!text.startsWith("<think>")) return text;
        int end = text.indexOf("</think>");
        if (end < 0) throw new IllegalArgumentException("Incomplete thinking block");
        return text.substring(end + "</think>".length()).stripLeading();
    }

    static List<Call> calls(String text) {
        if (!text.startsWith(OPEN)) return List.of();
        var calls = new ArrayList<Call>();
        while (!text.isEmpty()) {
            if (!text.startsWith(OPEN)) throw new IllegalArgumentException("Unexpected text after a tool call");
            int end = closingTag(text);
            if (end < 0) throw new IllegalArgumentException("Incomplete tool call");
            calls.add(call(text.substring(OPEN.length(), end)));
            text = text.substring(end + CLOSE.length()).stripLeading();
        }
        return calls;
    }

    static List<Call> interactiveTerminalCalls(String text) {
        int start = text.indexOf(OPEN);
        if (start < 0 || start > 0 && text.charAt(start - 1) != '\n') return List.of();
        String preamble = text.substring(0, start);
        if (preamble.contains("```") || preamble.contains("~~~")) return List.of();
        return calls(text.substring(start)).stream().map(ToolProtocol::normalizeTerminalArguments).toList();
    }

    private static Call normalizeTerminalArguments(Call call) {
        var arguments = call.arguments();
        if (call.name().equals("terminal") && arguments.size() == 1
                && arguments.get("command") instanceof List<?> command) {
            return new Call(call.name(), Map.of("argv", command));
        }
        return call;
    }

    private static int closingTag(String text) {
        boolean quoted = false;
        for (int index = OPEN.length(); index < text.length(); index++) {
            char character = text.charAt(index);
            if (quoted && character == '\\') {
                index++;
                continue;
            }
            if (character == '"') quoted = !quoted;
            if (!quoted && character == '<' && text.startsWith(CLOSE, index)) return index;
        }
        return -1;
    }

    private static Call call(String json) {
        var payload = Qwen3Json.object(Qwen3Json.parse(json));
        if (payload.size() != 2) throw new IllegalArgumentException("Expected tool name and arguments");
        return new Call(Qwen3Json.string(payload.get("name")), Qwen3Json.object(payload.get("arguments")));
    }

    static String response(String name, Map<String, Object> result) {
        return "<tool_response>\n" + json(Map.of("name", name, "result", result)) + "\n</tool_response>";
    }

    private static String json(Map<String, Object> data) {
        return Qwen3Json.stringify(data).replace("<", "\\u003c");
    }
}
