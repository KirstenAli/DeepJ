package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.data.StatefulBatchSource;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.TrainingProgress;
import java.io.*;
import java.nio.file.*;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

class NdTrainingCheckpointTest {
    @TempDir Path directory;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void allDtypesResumeNdLossGradientsWeightsMomentsAndUnroundedMastersExactly() throws IOException {
        for (DType dtype : DType.values()) {
            checkResume(dtype);
        }
    }

    private void checkResume(DType dtype) throws IOException {
        Parameter parameter = parameter(dtype);
        AdamW optimizer = AdamW.defaultAdamW(0.001f);
        var data = new Cursor();
        float loss = advance(parameter, optimizer, data, 13);
        var progress = new TrainingProgress(13, loss, loss);
        Path file = directory.resolve(dtype + ".dj");
        TrainingCheckpoint.save(List.of(parameter), optimizer, data, progress, schedule(), file);
        float expectedLoss = advance(parameter, optimizer, data, 7);
        byte[] expected = snapshot(parameter, optimizer, data, expectedLoss);
        assertEquals(progress, TrainingCheckpoint.load(List.of(parameter), optimizer, data, schedule(), file));
        assertEquals(13, optimizer.stepCount());
        assertEquals(13, data.randomState());
        float actualLoss = advance(parameter, optimizer, data, 7);
        assertArrayEquals(expected, snapshot(parameter, optimizer, data, actualLoss));
        assertEquals(dtype, parameter.value.dtype());
    }

    @Test
    void truncatedOrTrailingTypedFilesDoNotChangeModelOrOptimizerState() throws IOException {
        Parameter source = parameter(DType.FLOAT16), target = parameter(DType.FLOAT16);
        AdamW optimizer = AdamW.defaultAdamW(0.001f), restored = AdamW.defaultAdamW(0.001f);
        var data = new Cursor();
        float loss = advance(source, optimizer, data, 3);
        Path file = directory.resolve("damaged.dj");
        TrainingCheckpoint.save(List.of(source), optimizer, data,
                new TrainingProgress(3, loss, loss), schedule(), file);
        byte[] valid = Files.readAllBytes(file), before = snapshot(target, restored, new Cursor(), 0);
        Files.write(file, Arrays.copyOf(valid, valid.length - 1));
        assertRejected(target, restored, file, before);
        Files.write(file, Arrays.copyOf(valid, valid.length + 1));
        assertRejected(target, restored, file, before);
    }

    private static void assertRejected(Parameter target, AdamW optimizer, Path file, byte[] before) throws IOException {
        var data = new Cursor();
        assertThrows(IOException.class,
                () -> TrainingCheckpoint.load(List.of(target), optimizer, data, schedule(), file));
        assertArrayEquals(before, snapshot(target, optimizer, data, 0));
        assertEquals(0, target.value.gradientState().version());
    }

    @Test
    void hyperparameterMismatchIsDetectedBeforeApplyingModelValues() throws IOException {
        Parameter source = parameter(DType.BFLOAT16), target = parameter(DType.BFLOAT16);
        AdamW optimizer = AdamW.defaultAdamW(0.001f), mismatched = new AdamW(0.001f, 0.8f, 0.999f, 1e-8f, 0.01f);
        var data = new Cursor();
        float loss = advance(source, optimizer, data, 2);
        Path file = directory.resolve("hyperparameters.dj");
        TrainingCheckpoint.save(List.of(source), optimizer, data,
                new TrainingProgress(2, loss, loss), schedule(), file);
        float[] before = target.value.toFloatArray();
        assertThrows(IllegalArgumentException.class, () -> TrainingCheckpoint.load(
                List.of(target), mismatched, new Cursor(), schedule(), file));
        assertArrayEquals(before, target.value.toFloatArray());
        assertEquals(0, target.value.gradientState().version());
        assertEquals(0, mismatched.stepCount());
    }

    private static float advance(Parameter parameter, AdamW optimizer, Cursor data, int steps) {
        float loss = 0;
        for (int step = 0; step < steps; step++) {
            parameter.zeroGrad();
            int first = (int) (data.randomState() % 3);
            data.nextBatch(1);
            Tensor objective = CrossEntropyLoss.forward(parameter.value, new int[]{first, (first + 1) % 3}, null);
            loss = objective.item();
            assertTrue(Float.isFinite(loss));
            objective.backward();
            optimizer.step(List.of(parameter));
        }
        return loss;
    }

    private static byte[] snapshot(Parameter parameter, AdamW optimizer, Cursor data, float loss) throws IOException {
        var bytes = new ByteArrayOutputStream();
        var state = optimizer.state(List.of(parameter));
        try (var output = new DataOutputStream(bytes)) {
            output.writeFloat(loss);
            output.writeLong(state.step());
            output.writeLong(data.randomState());
            TensorEncoding.write(output, parameter.value);
            TensorEncoding.write(output, parameter.grad());
            TensorEncoding.write(output, state.firstMoments().get(0));
            TensorEncoding.write(output, state.secondMoments().get(0));
            if (!state.masterWeights().isEmpty()) TensorEncoding.write(output, state.masterWeights().get(0));
        }
        return bytes.toByteArray();
    }

    private static Parameter parameter(DType dtype) {
        return new Parameter(Tensor.from(new float[]{0.7f, -0.3f, 1.2f, 0.4f, -0.2f, 0.9f}, dtype, 2, 1, 3));
    }
    private static CosineLearningRateSchedule schedule() {
        return new CosineLearningRateSchedule(0.001f, 0.0001f, 2, 100);
    }

    private static final class Cursor implements StatefulBatchSource {
        private long cursor;
        @Override public Batch nextBatch(int size) {
            cursor++;
            return new Batch(new int[][]{{0}}, new int[][]{{1}});
        }
        @Override public long randomState() { return cursor; }
        @Override public void restoreRandomState(long state) { cursor = state; }
    }
}
