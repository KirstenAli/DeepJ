package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.nn.transformer.DecoderOnlyModel;
import io.github.kirstenali.deepj.nn.transformer.origin.*;
import io.github.kirstenali.deepj.nn.transformer.orbit.*;
import io.github.kirstenali.deepj.nn.transformer.prism.*;
import io.github.kirstenali.deepj.nn.transformer.nova.*;
import io.github.kirstenali.deepj.nn.transformer.loop.*;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import org.junit.jupiter.params.provider.Arguments;
import java.util.Arrays;
import java.util.stream.Stream;

/** Small architecture-complete cases,including compact KV groups and repeated Loop visits. */
public final class NdModelFixtures {
    private NdModelFixtures() {}
    public static Stream<String> families() { return Stream.of("origin", "orbit", "prism", "nova", "loop"); }

    public static Stream<Arguments> typedCases() {
        return families().flatMap(family -> Arrays.stream(DType.values()).map(dtype -> Arguments.of(family, dtype)));
    }

    public static DecoderOnlyModel model(String family, long seed, ParameterFactory parameters) {
        return switch (family) {
            case "origin" -> new DeepJOrigin(new DeepJOriginConfig(19, 8, 8, 2, 1, 16), seed, parameters);
            case "orbit" -> new DeepJOrbit(new DeepJOrbitConfig(19, 8, 8, 2, 1, 16), seed, parameters);
            case "prism" -> new DeepJPrism(new DeepJPrismConfig(19, 8, 8, 2, 1, 16, 4, 3), seed, parameters);
            case "nova" -> new DeepJNova(new DeepJNovaConfig(19, 8, 8, 2, 1, 1, 16), seed, parameters);
            case "loop" -> new DeepJLoop(new DeepJLoopConfig(19, 8, 8, 2, 1, 1, 1, 1, 2, 16), seed, parameters);
            default -> throw new IllegalArgumentException(family);
        };
    }

    public static DecoderOnlyModel model(String family, long seed) {
        return switch (family) {
            case "origin" -> new DeepJOrigin(new DeepJOriginConfig(19, 8, 8, 2, 1, 16), seed);
            case "orbit" -> new DeepJOrbit(new DeepJOrbitConfig(19, 8, 8, 2, 1, 16), seed);
            case "prism" -> new DeepJPrism(new DeepJPrismConfig(19, 8, 8, 2, 1, 16, 4, 3), seed);
            case "nova" -> new DeepJNova(new DeepJNovaConfig(19, 8, 8, 2, 1, 1, 16), seed);
            case "loop" -> new DeepJLoop(new DeepJLoopConfig(19, 8, 8, 2, 1, 1, 1, 1, 2, 16), seed);
            default -> throw new IllegalArgumentException(family);
        };
    }

    public static int[][] inputs() { return new int[][]{{1, 2, 1, 3}, {4, 4, 2, 1}}; }
    public static int[][] targets() { return new int[][]{{2, 1, 3, 5}, {4, 2, 1, 5}}; }
    public static boolean[][] masks() { return new boolean[][]{{false, false, false, true}, {true, true, true, true}}; }
}
