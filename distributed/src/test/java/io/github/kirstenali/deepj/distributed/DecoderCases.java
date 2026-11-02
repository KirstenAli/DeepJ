package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.nn.transformer.DecoderOnlyModel;
import io.github.kirstenali.deepj.nn.transformer.origin.*;
import io.github.kirstenali.deepj.nn.transformer.orbit.*;
import io.github.kirstenali.deepj.nn.transformer.prism.*;
import io.github.kirstenali.deepj.nn.transformer.nova.*;
import io.github.kirstenali.deepj.nn.transformer.loop.*;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import java.util.List;

final class DecoderCases {
    private DecoderCases() {}
    static List<String> families() { return List.of("origin", "orbit", "prism", "nova", "loop"); }

    static DecoderOnlyModel model(String family, DType dtype, long seed) {
        ParameterFactory parameters = new ParameterFactory(dtype);
        return switch (family) {
            case "origin" -> new DeepJOrigin(new DeepJOriginConfig(11, 8, 4, 1, 1, 6, 0.2f, 0.01f), seed, parameters);
            case "orbit" -> new DeepJOrbit(new DeepJOrbitConfig(11, 8, 4, 1, 1, 6, 0.2f, 0.01f), seed, parameters);
            case "prism" -> new DeepJPrism(new DeepJPrismConfig(11, 8, 4, 1, 1, 6, 2, 2, 0.2f, 0.01f),
                    seed, parameters);
            case "nova" -> new DeepJNova(new DeepJNovaConfig(11, 8, 4, 1, 1, 1, 6, true, 0.2f, 0.01f),
                    seed, parameters);
            case "loop" -> new DeepJLoop(new DeepJLoopConfig(11, 8, 4, 1, 1, 1, 1, 1, 2, 6,
                    true, 0.2f, 0.01f, 0.5f), seed, parameters);
            default -> throw new IllegalArgumentException(family);
        };
    }

    static Batch data() {
        int[][] inputs = new int[7][4], targets = new int[7][4];
        boolean[][] masks = new boolean[7][4];
        for (int row = 0; row < inputs.length; row++) {
            for (int time = 0; time < inputs[row].length; time++) {
                inputs[row][time] = (row + time) % 11;
                targets[row][time] = (row + time + 1) % 11;
                masks[row][time] = time >= row % 3;
            }
        }
        return new Batch(inputs, targets, masks);
    }
}
