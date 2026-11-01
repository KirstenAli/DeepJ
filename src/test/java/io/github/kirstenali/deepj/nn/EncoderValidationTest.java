package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import java.util.Random;
import static io.github.kirstenali.deepj.nn.NeuralTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class EncoderValidationTest extends NeuralTest {
    @Test void malformedEncoderConfigurationIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new TransformerEncoderConfig(0, 1, 2, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> new TransformerEncoderConfig(2, 0, 2, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> new TransformerEncoderConfig(2, 1, 0, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> new TransformerEncoderConfig(2, 1, 2, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> new TransformerEncoderConfig(3, 2, 4, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> new TransformerEncoderConfig(2, 1, 2, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new EncoderAttention(3, 2, 0,
                new Random(1), ParameterFactory.float32()));
    }

    @Test void maskDimensionsAndFullyMaskedSamplesAreRejected() {
        TransformerEncoder encoder = new TransformerEncoder(new TransformerEncoderConfig(2, 1, 3, 1, 0), new Random(1));
        Tensor input = values(1, 2, 2);
        assertThrows(IllegalArgumentException.class, () -> encoder.forward(input, new boolean[2][2]));
        assertThrows(IllegalArgumentException.class, () -> encoder.forward(input, new boolean[1][1]));
        assertThrows(IllegalArgumentException.class, () -> encoder.forward(input, new boolean[][]{null}));
        assertThrows(IllegalArgumentException.class, () -> encoder.forward(input, new boolean[][]{{false, false}}));
        assertThrows(IllegalArgumentException.class, () -> encoder.forward(values(1, 2, 3)));
        assertThrows(IllegalArgumentException.class, () -> encoder.forward(values(2, 2)));
    }
}
