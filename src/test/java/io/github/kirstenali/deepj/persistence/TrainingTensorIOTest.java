package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TrainingTensorIOTest {
    @TempDir Path directory;

    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void validationKeepsOnlyWeightOffsetsAndRestoresEveryDtypeThroughTheSameReader() throws IOException {
        for (DType dtype : DType.values()) { check(dtype); }
    }

    private void check(DType dtype) throws IOException {
        var source = List.of(parameter(dtype, new float[]{-0.0f, 2, 3, 4}),
                parameter(dtype, new float[]{5, 6, 7, 8}));
        var target = List.of(parameter(dtype, new float[4]), parameter(dtype, new float[4]));
        Path path = write(source, dtype);
        try (var file = new CheckpointInput(path)) {
            assertEquals(2, file.data().readInt());
            var state = TrainingTensorIO.read(file, target, true);
            assertEquals(file.size(), file.position());
            assertEquals(2, state.modelPositions().size());
            assertTrue(state.modelPositions().get(1) > state.modelPositions().get(0));
            assertArrayEquals(new float[4], target.get(0).value.toFloatArray());
            restoreAndCompare(file, source, target, state);
        }
    }

    private static void restoreAndCompare(CheckpointInput file, List<Parameter> source,
                                           List<Parameter> target, TrainingTensorIO.State state) throws IOException {
        TrainingTensorIO.apply(target, file, state.modelPositions(), true);
        for (int index = 0; index < source.size(); index++) {
            assertArrayEquals(source.get(index).value.toFloatArray(), target.get(index).value.toFloatArray());
            assertEquals(DType.FLOAT32, state.first().get(index).dtype());
            assertArrayEquals(new float[4], state.second().get(index).toFloatArray());
        }
        assertEquals(source.get(0).value.dtype() == DType.FLOAT32 ? 0 : 2, state.masters().size());
    }

    private Path write(List<Parameter> parameters, DType dtype) throws IOException {
        Path path = directory.resolve(dtype + ".bin");
        try (var output = new DataOutputStream(Files.newOutputStream(path))) {
            TrainingTensorIO.write(output, parameters, AdamW.defaultAdamW(1e-4f).state(parameters), true);
        }
        return path;
    }

    private static Parameter parameter(DType dtype, float[] values) {
        return new Parameter(Tensor.from(values, dtype, 2, 2));
    }
}
