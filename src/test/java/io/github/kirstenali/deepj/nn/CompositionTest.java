package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.activations.Tanh;
import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import java.util.Random;
import static io.github.kirstenali.deepj.nn.NeuralTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class CompositionTest extends NeuralTest {
    @Test void existingLayersComposeWithNdShapesAndIdentityDeduplication() {
        Linear shared = new Linear(2, 2, new Random(1));
        Sequential network = new Sequential(shared, Activation.tanh(), shared);
        assertEquals(2, network.parameters().size());
        assertEquals(3, network.layers().size());
        initialize(network);
        gradients(network, values(2, 2, 2));
        assertThrows(UnsupportedOperationException.class, () -> network.layers().clear());
    }

    @Test void flattenAndReshapePreserveOrderingAndGradientsThroughStridedViews() {
        Tensor input = values(2, 3, 4).transpose(1, 2).requiresGrad(true);
        Tensor output = new Sequential(new Flatten(), new Reshape(-1, 4, 3)).forward(input);
        assertArrayEquals(new int[]{2, 4, 3}, output.shape());
        assertArrayEquals(input.toFloatArray(), output.toFloatArray());
        output.sumTensor().backward();
        for (float gradient : input.grad().toFloatArray()) { assertEquals(1, gradient); }
        assertArrayEquals(new int[]{24}, new Flatten(0, -1).forward(input).shape());
        assertArrayEquals(new int[]{2, 4, 3}, new Flatten(1, 1).forward(input).shape());
    }

    @Test void adaptersReuseExistingActivationDerivatives() {
        for (Activation activation : new Activation[]{Activation.relu(), Activation.gelu(), Activation.sigmoid(),
                Activation.tanh(), Activation.softmax(-1), new Activation(new Tanh())}) {
            gradients(activation, Tensor.from(new float[]{-0.7f, 0.3f, 0.8f}, 1, 3));
        }
    }

    @Test void mlpAndAutoencoderReuseFnnAndParameterFactories() {
        MLP encoder = new MLP(3, new int[]{4}, 2, new Random(2));
        MLP decoder = new MLP(2, new int[0], 3, Tanh::new, null, new Random(3), ParameterFactory.float32());
        Autoencoder network = new Autoencoder(encoder, decoder);
        initialize(network);
        assertArrayEquals(new int[]{2, 2}, network.encode(values(2, 3)).shape());
        assertArrayEquals(new int[]{2, 3}, network.decode(values(2, 2)).shape());
        assertEquals(6, network.parameters().size());
        gradients(network, values(2, 3));
    }

    @Test void malformedShapesAndConfigurationsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Reshape(-1, -1));
        assertThrows(IllegalArgumentException.class, () -> new Reshape(0));
        assertThrows(IllegalArgumentException.class, () -> new Reshape(-1, 3).forward(values(2, 2)));
        assertThrows(IllegalArgumentException.class, () -> new Flatten(2, 1).forward(values(2, 2, 2)));
        assertThrows(IllegalArgumentException.class, () -> new Dropout(Float.NaN, 0));
        assertThrows(IllegalArgumentException.class, () -> new Dropout(-0.1f, 0));
        assertThrows(IllegalArgumentException.class, () -> new Dropout(1, 0));
        assertThrows(NullPointerException.class, () -> new Autoencoder(null, Activation.relu()));
        assertEquals(4, new Reshape(2, 2).forward(values(4)).numel());
        assertEquals(1, new Reshape().forward(values(1)).numel());
    }

    @Test void modeChangesPropagateToNestedNetworksAndCompactMlpWorks() {
        Dropout dropout = new Dropout(0, 1);
        Sequential network = new Sequential(new Autoencoder(dropout, new Sequential(Activation.relu())));
        network.eval();
        assertFalse(dropout.isTraining());
        network.train();
        assertTrue(dropout.isTraining());
        valuesIdentity(dropout);
        for (DType dtype : DType.values()) {
            MLP mlp = new MLP(2, new int[]{3}, 1, Tanh::new, null, new Random(2), new ParameterFactory(dtype));
            assertArrayEquals(new int[]{2, 1}, mlp.forward(values(2, 2).to(dtype)).shape());
        }
    }

    private static Tensor valuesIdentity(Dropout dropout) {
        Tensor input = Tensor.ones(1, 1);
        assertSame(input, dropout.forward(input));
        return input;
    }
}
