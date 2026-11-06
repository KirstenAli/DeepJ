package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tokenizers.gemma.GemmaTokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

/** Exports exact Java prompt IDs and initial noise; Python independently computes all expected outputs. */
public final class SanaVideoFixtureInputs {
    private SanaVideoFixtureInputs() {}

    public static void main(String[] arguments) throws Exception {
        Path root = Path.of("src/test/resources/sana-video");
        var tokenizer = new GemmaTokenizer(root.resolve("pipeline/tokenizer"));
        var positive = new SanaVideoPrompt(tokenizer, "a fox", true);
        var negative = new SanaVideoPrompt(tokenizer, "", false);
        var input = Map.of("positive", caption(positive), "negative", caption(negative), "noise", noise(),
                "prompt", "a fox", "width", 16, "height", 16, "frames", 9, "steps", 5, "seed", 123, "guidance", 6);
        Files.writeString(root.resolve("pipeline-inputs.json"), Qwen3Json.stringify(input) + "\n");
    }

    private static Map<String, Object> caption(SanaVideoPrompt prompt) {
        return Map.of("tokens", integers(prompt.tokens()), "valid", integers(prompt.valid()),
                "selected", integers(prompt.selected()), "mask", integers(prompt.mask()));
    }

    private static List<Integer> integers(int[] values) { return Arrays.stream(values).boxed().toList(); }

    private static List<Float> noise() {
        var random = new Random(123);
        List<Float> values = new ArrayList<>();
        for (int i = 0; i < 16 * 3 * 2 * 2; i++) { values.add((float) random.nextGaussian()); }
        return values;
    }
}
