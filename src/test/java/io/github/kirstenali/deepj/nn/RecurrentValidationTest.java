package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.DType;
import org.junit.jupiter.api.Test;
import java.util.Random;
import static io.github.kirstenali.deepj.nn.NeuralTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class RecurrentValidationTest extends NeuralTest {
    @Test void malformedLengthsAndSequenceShapesAreRejected() {
        Recurrent recurrent = new RNN(2, 2, new Random(1));
        assertThrows(IllegalArgumentException.class, () -> recurrent.run(values(2, 2, 2), null, new int[]{1}));
        assertThrows(IllegalArgumentException.class, () -> recurrent.run(values(1, 2, 2), null, new int[]{-1}));
        assertThrows(IllegalArgumentException.class, () -> recurrent.run(values(1, 2, 2), null, new int[]{3}));
        assertThrows(IllegalArgumentException.class, () -> recurrent.forward(values(2, 2)));
        assertThrows(IllegalArgumentException.class, () -> recurrent.forward(values(2, 2, 3)));
        assertThrows(IllegalArgumentException.class, () -> new RNNCell(0, 2, new Random(1)));
        assertThrows(IllegalArgumentException.class, () -> new RNNCell(2, 0, new Random(1)));
    }

    @Test void malformedHiddenAndCellStatesAreRejected() {
        RecurrentCell rnn = new RNNCell(2, 2, new Random(1)), lstm = new LSTMCell(2, 2, new Random(1));
        assertThrows(IllegalArgumentException.class, () -> rnn.step(values(1, 2), null));
        assertThrows(IllegalArgumentException.class, () -> rnn.step(values(1, 3), new RecurrentState(values(1, 2))));
        assertThrows(IllegalArgumentException.class, () -> rnn.step(values(1, 2), new RecurrentState(values(2, 2))));
        assertThrows(IllegalArgumentException.class, () -> rnn.step(values(1, 2), new RecurrentState(values(1, 3))));
        assertThrows(IllegalArgumentException.class, () -> rnn.step(values(1, 2), new RecurrentState(values(2))));
        assertThrows(IllegalArgumentException.class, () -> lstm.step(values(1, 2), new RecurrentState(values(1, 2))));
        assertThrows(IllegalArgumentException.class, () -> rnn.step(values(1, 2),
                new RecurrentState(values(1, 2), values(1, 2))));
        assertThrows(IllegalArgumentException.class, () -> rnn.initialState(0, DType.FLOAT32));
    }
}
