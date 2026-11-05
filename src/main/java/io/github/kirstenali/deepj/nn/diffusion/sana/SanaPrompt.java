package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.tokenizers.gemma.GemmaTokenizer;
import java.util.Arrays;
import java.util.Locale;

/** The official SANA-Sprint instruction prefix and BOS-plus-tail prompt selection.
 * Prefix: Copyright 2024 PixArt-Sigma Authors and The HuggingFace Team; Apache-2.0.
 * See docs/third-party/sana-diffusers-license.txt. */
final class SanaPrompt {
    private static final String PREFIX =
            "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual " +
            "descriptions suitable for image generation. Evaluate the level of detail in the user" +
            " prompt:\n" +
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, " +
            "textures, and spatial relationships to create vivid and concrete scenes.\n" +
            "- If the prompt is already detailed, refine and enhance the existing details " +
            "slightly without overcomplicating.\n" +
            "Here are examples of how to transform or refine prompts:\n" +
            "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a " +
            "round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of " +
            "blooming red flowers.\n" +
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk," +
            " featuring glowing street lamps, a diverse crowd of people in colorful clothing, and" +
            " a double-decker bus passing by towering glass skyscrapers.\n" +
            "Please generate only the enhanced description for the prompt below and avoid " +
            "including any additional commentary or evaluations:\n" +
            "User Prompt: ";
    private final int[] tokens, valid, selected;

    SanaPrompt(GemmaTokenizer tokenizer, String prompt) {
        int length = tokenizer.encode(PREFIX).length + 298;
        int[] encoded = tokenizer.encode(PREFIX + prompt.toLowerCase(Locale.ROOT).strip());
        tokens = Arrays.copyOf(encoded, length);
        valid = new int[length];
        Arrays.fill(valid, 0, Math.min(encoded.length, length), 1);
        selected = java.util.stream.IntStream.range(0, 300).map(i -> i == 0 ? 0 : length - 300 + i).toArray();
    }

    int[] tokens() { return tokens.clone(); }
    int[] valid() { return valid.clone(); }
    int[] selected() { return selected.clone(); }
    int[] mask() { return Arrays.stream(selected).map(i -> valid[i]).toArray(); }
}
