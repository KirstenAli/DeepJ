package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ModelSerializerMetalTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void loadDiscardsPendingMetalOperations() throws Exception {
        Assumptions.assumeTrue(MetalBackend.isAvailable());
        Path checkpoint = saveCheckpoint();
        TensorBackend previous = Tensor.backend();
        Tensor.setBackend(new MetalBackend());
        try {
            Parameter target = pendingParameter();
            ModelSerializer.load(List.of(target), checkpoint);
            assertEquals(7.0f, target.value.multiplyScalar(1.0f).get(0, 0), 0.0f);
        } finally {
            Tensor.backend().releaseResources();
            Tensor.setBackend(previous);
        }
    }

    private Path saveCheckpoint() throws Exception {
        TensorBackend previous = Tensor.backend();
        Tensor.setBackend(new CpuBackend());
        try {
            Path path = temporaryDirectory.resolve("model.dj");
            ModelSerializer.save(List.of(new Parameter(Tensor.from2D(new float[][]{{7.0f}}))), path);
            return path;
        } finally {
            Tensor.setBackend(previous);
        }
    }

    private static Parameter pendingParameter() {
        Parameter parameter = new Parameter(Tensor.from2D(new float[][]{{2.0f}}));
        parameter.value.multiplyScalarInPlace(2.0f);
        return parameter;
    }
}
