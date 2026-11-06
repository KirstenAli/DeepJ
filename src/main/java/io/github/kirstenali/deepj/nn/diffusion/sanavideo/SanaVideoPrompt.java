// Portions adapted from Diffusers (Apache-2.0); see META-INF/licenses/sana-video-NOTICE.txt.
package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import java.util.Arrays;
import java.util.Locale;
import java.util.stream.IntStream;

/** Diffusers SANA-Video prompt instruction and BOS-plus-tail selection (Apache-2.0; see docs/sana-video.md). */
final class SanaVideoPrompt {
    static final int LENGTH = 300;
    static final String INSTRUCTION = String.join("\n",
            "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable "
                    + "for video generation. Evaluate the level of detail in the user prompt:",
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, motion, "
                    + "and temporal relationships to create vivid and dynamic scenes.",
            "- If the prompt is already detailed, refine and enhance the existing details slightly without "
                    + "overcomplicating.",
            "Here are examples of how to transform or refine prompts:",
            "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat slowly settling into a curled "
                    + "position, peacefully falling asleep on a warm sunny windowsill, with gentle sunlight "
                    + "filtering through surrounding pots of blooming red flowers.",
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing "
                    + "street lamps gradually lighting up, a diverse crowd of people in colorful clothing walking "
                    + "past, and a double-decker bus smoothly passing by towering glass skyscrapers.",
            "Please generate only the enhanced description for the prompt below and avoid including any additional "
                    + "commentary or evaluations:",
            "User Prompt: ");
    private final int[] tokens;
    private final int[] valid;
    private final int[] selected;

    SanaVideoPrompt(Tokenizer tokenizer, String text, boolean positive) {
        String prefix = positive ? INSTRUCTION : "";
        int length = positive ? tokenizer.encode(prefix).length + LENGTH - 2 : LENGTH;
        int[] encoded = tokenizer.encode(prefix + text.toLowerCase(Locale.ROOT).strip());
        tokens = Arrays.copyOf(encoded, length);
        valid = new int[length];
        Arrays.fill(valid, 0, Math.min(length, encoded.length), 1);
        selected = IntStream.range(0, LENGTH).map(i -> i == 0 ? 0 : length - LENGTH + i).toArray();
    }

    int[] tokens() { return tokens.clone(); }
    int[] valid() { return valid.clone(); }
    int[] selected() { return selected.clone(); }
    int[] mask() { return Arrays.stream(selected).map(i -> valid[i]).toArray(); }
}
