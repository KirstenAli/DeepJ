package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOrigin;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterOptimizer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GradientAccumulationTest {

    @Test
    void twoMicroBatchesMatchOneBatchOfTwo() throws IOException {
        Recorder plain = new Recorder(), accumulated = new Recorder();
        CausalLMTraining.trainer(model(), dataset(), plain).trainStep(2);
        GradientAccumulation.trainer(model(), dataset(), accumulated, 2).trainStep(1);
        assertEquals(plain.steps, accumulated.steps);
        assertGradientsClose(plain.gradients, accumulated.gradients);
    }

    @Test
    void lossIsAveragedOverAllSamples() throws IOException {
        List<Float> single = new ArrayList<>();
        Trainer one = CausalLMTraining.trainer(model(), dataset(), new Recorder());
        for (int step = 0; step < 2; step++) { single.add(one.trainStep(1)); }
        float accumulated = GradientAccumulation.trainer(model(), dataset(), new Recorder(), 2).trainStep(1);
        assertEquals((single.get(0) + single.get(1)) * 0.5f, accumulated, 1e-5f);
    }

    @Test
    void rejectsInvalidArguments() throws IOException {
        TextDataset data = dataset();
        assertThrows(IllegalArgumentException.class, () -> GradientAccumulation.trainer(model(), data, 1e-2f, 0));
        assertThrows(IllegalArgumentException.class, () -> GradientAccumulation.trainer(null, data, 1e-2f, 1));
    }

    private static DeepJOrigin model() {
        var config = new DeepJOriginConfig(ByteTokenizer.VOCAB_SIZE, 8, 32, 4, 1, 64, 0.2f, Float.MAX_VALUE);
        return new DeepJOrigin(config, 3L);
    }

    private static TextDataset dataset() throws IOException {
        Path tmp = Files.createTempFile("deepj_accum", ".txt");
        Files.writeString(tmp, "one two three four five six seven eight nine ten");
        return TextDataset.fromFile(tmp, new ByteTokenizer(), 8, 1L);
    }

    private static void assertGradientsClose(List<Tensor> expected, List<Tensor> actual) {
        assertEquals(expected.size(), actual.size());
        for (int index = 0; index < expected.size(); index++) {
            assertArrayEquals(expected.get(index).rawData(), actual.get(index).rawData(), 1e-6f);
        }
    }

    private static final class Recorder implements ParameterOptimizer {
        final List<Tensor> gradients = new ArrayList<>();
        int steps;

        @Override
        public void step(List<Parameter> parameters) {
            steps++;
            parameters.forEach(parameter -> gradients.add(new Tensor(parameter.grad())));
        }
    }
}

