package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.data.StatefulBatchSource;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.TrainingProgress;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.file.*;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

class TypedCheckpointCorruptionTest {
    @TempDir Path directory;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void modelMetadataFlagsAndParameterCountsAreStrictlyValidated() throws IOException {
        Path file = directory.resolve("model.dj");
        Parameter parameter = parameter(DType.FLOAT16);
        ModelSerializer.save(List.of(parameter), file);
        byte[] valid = Files.readAllBytes(file), corrupt = valid.clone();
        corrupt[8] = 2;
        Files.write(file, corrupt);
        assertThrows(IOException.class, () -> ModelSerializer.settings(file));
        ByteBuffer.wrap(valid).putInt(9, -1);
        Files.write(file, valid);
        assertThrows(IOException.class, () -> ModelSerializer.load(List.of(parameter), file));
        assertEquals(0, parameter.value.gradientState().version());
    }

    @Test
    void missingUnexpectedAndNonBooleanMasterFlagsCannotRestorePartialState() throws IOException {
        for (DType dtype : new DType[]{DType.FLOAT32, DType.FLOAT16}) {
            Path file = directory.resolve(dtype + ".dj");
            Parameter parameter = parameter(dtype);
            AdamW optimizer = AdamW.defaultAdamW(0.001f);
            TrainingCheckpoint.save(List.of(parameter), optimizer, new EmptyData(),
                    new TrainingProgress(0, 1, 1), schedule(), file);
            byte[] valid = Files.readAllBytes(file);
            int flag = masterFlagOffset(valid);
            byte[] corrupt = valid.clone();
            corrupt[flag] = (byte) (dtype == DType.FLOAT32 ? 1 : 0);
            assertRejected(file, corrupt, parameter, optimizer);
            corrupt[flag] = 2;
            assertRejected(file, corrupt, parameter, optimizer);
        }
    }

    private static int masterFlagOffset(byte[] bytes) throws IOException {
        try (var input = new DataInputStream(new ByteArrayInputStream(bytes))) {
            // Version3's unconfigured fixed metadata ends at the parameter count at byte73.
            input.skipNBytes(73);
            assertEquals(1, input.readInt());
            for (int tensor = 0; tensor < 3; tensor++) {
                input.skipNBytes(TensorDescriptor.read(input).bytes());
            }
            return bytes.length - input.available();
        }
    }

    private static void assertRejected(Path file, byte[] bytes, Parameter parameter, AdamW optimizer)
            throws IOException {
        Files.write(file, bytes);
        assertThrows(IOException.class, () -> TrainingCheckpoint.load(
                List.of(parameter), optimizer, new EmptyData(), schedule(), file));
        assertArrayEquals(new float[]{1, 2}, parameter.value.toFloatArray());
        assertEquals(0, parameter.value.gradientState().version());
        assertEquals(0, optimizer.stepCount());
    }

    private static Parameter parameter(DType dtype) {
        return new Parameter(Tensor.from(new float[]{1, 2}, dtype));
    }
    private static CosineLearningRateSchedule schedule() {
        return new CosineLearningRateSchedule(0.001f, 0.0001f, 2, 100);
    }

    private static final class EmptyData implements StatefulBatchSource {
        @Override public Batch nextBatch(int size) { throw new UnsupportedOperationException(); }
        @Override public long randomState() { return 0; }
        @Override public void restoreRandomState(long state) { assertEquals(0, state); }
    }
}
