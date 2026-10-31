package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.Autograd;
import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

class TensorScopeCallbackTest {
    @Test
    void customBackwardRuleCanKeepPublicViewsBeyondLibraryScopes() {
        var saved = new AtomicReference<Tensor>();
        try (var ignored = TensorScope.open()) {
            Tensor input = Tensor.ones(1, 1).requiresGrad(true);
            Tensor output = Autograd.record(Tensor.ones(1, 1), gradient -> {
                assertNull(TensorScope.current());
                saved.set(Tensor.aliasStorage(gradient));
                return new Tensor[]{gradient};
            }, input);
            output.backward();
            assertEquals(1, input.grad().item());
        }
        assertFalse(saved.get().memory().canReleaseTemporary());
    }

    @Test
    void callerSuppliedGenerationFunctionDoesNotInheritPrivateOwnership() {
        var saved = new AtomicReference<Tensor>();
        try (var outer = TensorScope.open()) {
            TextGenerator.generate(ids -> {
                assertNull(TensorScope.current());
                Tensor logits = Tensor.aliasStorage(Tensor.ones(1, ByteTokenizer.VOCAB_SIZE));
                saved.set(logits);
                return logits;
            }, 4, new ByteTokenizer(), "a", 1, 1, 1, 2L);
            assertSame(outer, TensorScope.current());
        }
        assertFalse(saved.get().memory().canReleaseTemporary());
    }
}
