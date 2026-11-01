package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.nn.transformer.*;
import io.github.kirstenali.deepj.nn.transformer.origin.*;
import io.github.kirstenali.deepj.nn.transformer.orbit.*;
import io.github.kirstenali.deepj.nn.transformer.prism.*;
import io.github.kirstenali.deepj.nn.transformer.nova.*;
import io.github.kirstenali.deepj.nn.transformer.loop.*;
import io.github.kirstenali.deepj.tensor.DType;
import java.io.IOException;
import java.nio.file.Path;

/** Byte-tokenizer fixtures exercise public chat frontends,not merely direct logit calls. */
public final class TypedChatFixtures {
    private TypedChatFixtures() {}

    public static DecoderOnlyModel model(String family, DType dtype) {
        return switch (family) {
            case "origin" -> new DeepJOrigin(new DeepJOriginConfig(256, 64, 8, 2, 1, 16), 42, dtype);
            case "orbit" -> new DeepJOrbit(new DeepJOrbitConfig(256, 64, 8, 2, 1, 16), 42, dtype);
            case "prism" -> new DeepJPrism(new DeepJPrismConfig(256, 64, 8, 2, 1, 16, 4, 3), 42, dtype);
            case "nova" -> new DeepJNova(new DeepJNovaConfig(256, 64, 8, 2, 1, 1, 16), 42, dtype);
            case "loop" -> new DeepJLoop(new DeepJLoopConfig(256, 64, 8, 2, 1, 1, 1, 1, 2, 16), 42, dtype);
            default -> throw new IllegalArgumentException(family);
        };
    }

    public static Path save(String family, DType dtype, Path directory) throws IOException {
        DecoderOnlyModel model = model(family, dtype);
        model.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        Path path = directory.resolve("model.dj");
        model.save(path);
        return path;
    }
}
