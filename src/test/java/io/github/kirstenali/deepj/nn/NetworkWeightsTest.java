package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.activations.Tanh;
import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import static io.github.kirstenali.deepj.nn.NeuralTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class NetworkWeightsTest extends NeuralTest {
    @TempDir Path directory;

    @Test void weightsAndRunningBuffersRoundTripForEveryDtypeWithoutChangingTheExistingFormats() throws Exception {
        for (DType dtype : DType.values()) {
            Sequential source = model(dtype, 1), restored = model(dtype, 2);
            source.forward(values(2, 3));
            source.eval();
            source.forward(values(2, 3)).sumTensor().backward();
            Path path = directory.resolve(dtype + ".djnn");
            NetworkWeights.save(path, source);
            NetworkWeights.load(path, restored);
            restored.eval();
            assertArrayEquals(source.forward(values(2, 3)).toFloatArray(),
                    restored.forward(values(2, 3)).toFloatArray());
            assertArrayEquals(source.buffers().get(0).toFloatArray(), restored.buffers().get(0).toFloatArray());
            NetworkWeights.save(path, restored);
        }
    }

    @Test void failedLoadsLeaveEveryParameterAndBufferUntouched() throws Exception {
        Sequential source = model(DType.FLOAT32, 1), target = model(DType.FLOAT32, 2);
        Path valid = directory.resolve("valid.djnn");
        NetworkWeights.save(valid, source);
        byte[] bytes = Files.readAllBytes(valid);
        unchangedAfterFailure(target, Arrays.copyOf(bytes, bytes.length - 1));
        byte[] trailing = Arrays.copyOf(bytes, bytes.length + 1);
        unchangedAfterFailure(target, trailing);
        byte[] badMagic = bytes.clone();
        badMagic[0] = 0;
        unchangedAfterFailure(target, badMagic);
        byte[] badVersion = bytes.clone();
        ByteBuffer.wrap(badVersion).putInt(4, 999);
        unchangedAfterFailure(target, badVersion);
    }

    @Test void incompatibleNetworksNamesShapesAndDtypesAreRejected() throws Exception {
        Path path = directory.resolve("model.djnn");
        NetworkWeights.save(path, new Linear(2, 3, new Random(1)));
        assertThrows(IOException.class, () -> NetworkWeights.load(path, new Sequential()));
        assertThrows(IOException.class, () -> NetworkWeights.load(path, new Linear(3, 3, new Random(1))));
        assertThrows(IOException.class, () -> NetworkWeights.load(path,
                new Linear(2, 3, new Random(1), new ParameterFactory(DType.FLOAT16))));
        assertThrows(IOException.class, () -> NetworkWeights.load(path,
                new Linear(2, 3, new Random(1), ParameterFactory.float32().scope("different"))));
        NetworkWeights.save(path, new Sequential(new Linear(2, 2, new Random(1))));
        assertThrows(IOException.class, () -> NetworkWeights.load(path, new Sequential()));
    }

    @Test void sharedLayersAndBuffersAreWrittenOnlyOnce() throws Exception {
        BatchNorm norm = new BatchNorm(2);
        Sequential network = new Sequential(norm, norm);
        Path path = directory.resolve("shared.djnn");
        network.forward(values(2, 2));
        NetworkWeights.save(path, network);
        NetworkWeights.load(path, network);
        assertEquals(2, network.parameters().size());
        assertEquals(4, network.buffers().size());
        try (var files = Files.list(directory)) { assertEquals(1, files.count()); }
    }

    private void unchangedAfterFailure(Sequential target, byte[] bytes) throws Exception {
        Path path = directory.resolve("invalid.djnn");
        Files.write(path, bytes);
        float[] before = target.parameters().get(0).value.toFloatArray();
        float[] buffer = target.buffers().get(0).toFloatArray();
        assertThrows(IOException.class, () -> NetworkWeights.load(path, target));
        assertArrayEquals(before, target.parameters().get(0).value.toFloatArray());
        assertArrayEquals(buffer, target.buffers().get(0).toFloatArray());
    }

    private static Sequential model(DType dtype, long seed) {
        return new Sequential(new FNN(3, new int[]{4}, 2, Tanh::new, null, new Random(seed),
                new ParameterFactory(dtype)), new BatchNorm(2), Activation.tanh());
    }
}
