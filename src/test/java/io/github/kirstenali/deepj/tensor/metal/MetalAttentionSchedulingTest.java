package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.AttentionGradients;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

@Timeout(30)
class MetalAttentionSchedulingTest extends MetalDifferentialTestSupport {
    private final List<Tensor> buffers = new ArrayList<>();

    @BeforeEach
    void requireBlockAttention() { assumeTrue(MetalNative.TILED_ATTENTION_AVAILABLE); }

    @ParameterizedTest
    @ValueSource(ints = {1, 97, 193})
    void longGraphsDrainTheirBoundedQueuesAndPreserveResults(int stages) {
        int[] command = command();
        float[] expected = execute(command);
        float[] actual = execute(repeat(command, stages));
        assertArrayEquals(expected, actual);
    }

    @Test
    void batchDrainingBetweenBackwardPhasesPreservesAllThreeGradients() {
        Tensor q = random(1024, 64, 277), k = random(1024, 64, 281);
        Tensor v = random(1024, 64, 283), gradient = random(1024, 64, 293);
        var first = metal.causalAttention(q, k, v, 1, 0.125f);
        List<Tensor> expected = copies(metal.causalAttentionBackward(q, k, v, first, gradient, 1, 0.125f));
        var saved = metal.causalAttention(q, k, v, 1, 0.125f);
        var actual = metal.causalAttentionBackward(q, k, v, saved, gradient, 1, 0.125f);
        for (int repeat = 1; repeat < 100; repeat++) {
            actual = metal.causalAttentionBackward(q, k, v, saved, gradient, 1, 0.125f);
        }
        List<Tensor> values = copies(actual);
        for (int index = 0; index < values.size(); index++) {
            assertArrayEquals(expected.get(index).data, values.get(index).data);
        }
    }

    private static List<Tensor> copies(AttentionGradients gradients) {
        return List.of(new Tensor(gradients.query()), new Tensor(gradients.key()), new Tensor(gradients.value()));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 97})
    void failedEncodingDoesNotLeaveGpuJobsWaitingForAnUnsubmittedGraph(int stages) {
        int[] command = command();
        float[] expected = execute(command);
        int[] valid = repeat(command, stages);
        int[] failed = Arrays.copyOf(valid, valid.length + 10);
        System.arraycopy(new int[]{ComputeGraph.OP_ADD, command[1], command[1], command[4], 1024 * 64,
                ComputeGraph.OP_ADD, -1, command[1], command[4], 1024 * 64}, 0, failed, valid.length, 10);
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushOps(failed, failed.length));
        assertArrayEquals(expected, execute(command));
    }

    private int[] command() {
        Tensor input = random(1024, 64, 271), output = Tensor.zeros(1024, 64);
        Tensor normalization = Tensor.zeros(1024, 2);
        buffers.addAll(List.of(input, output, normalization));
        int inputId = metal.graph.ensureGpuBuffer(input).id;
        int outputId = metal.graph.ensureGpuBuffer(output).id;
        int normalizationId = metal.graph.ensureGpuBuffer(normalization).id;
        metal.graph.flush();
        return new int[]{ComputeGraph.OP_TILED_CAUSAL_ATTENTION, inputId, inputId, inputId,
                outputId, normalizationId, 1024, 1, Float.floatToRawIntBits(0.125f)};
    }

    private static int[] repeat(int[] command, int count) {
        int[] commands = new int[command.length * count];
        for (int index = 0; index < count; index++) {
            System.arraycopy(command, 0, commands, index * command.length, command.length);
        }
        return commands;
    }

    private static float[] execute(int[] commands) {
        MetalNative.nativeFlushOps(commands, commands.length);
        float[] output = new float[1024 * 64];
        MetalNative.nativeDownloadBuffer(commands[4], output);
        return output;
    }
}
