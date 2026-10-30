package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BackwardValidationOrderTest {
    @BeforeEach
    void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void everyInputAndSavedVersionIsCheckedForRootsAndSharedViews(DType dtype) {
        for (boolean aliases : new boolean[]{false, true}) {
            for (int count = 1; count <= 6; count++) {
                for (int index = 0; index < count; index++) {
                    checkMutation(dtype, aliases, count, index, false);
                    checkMutation(dtype, aliases, count, index, true);
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void emptyInputAndSavedListsRemainValid(DType dtype) {
        Fixture fixture = fixture(dtype, 0, false);
        assertDoesNotThrow(fixture.node()::validate);
        fixture.node().release();
    }

    @Test
    void inputVersionFailurePrecedesSavedVersionFailure() {
        Fixture fixture = fixture(DType.FLOAT32, 3, true);
        fixture.inputs()[1].gradientState().beforeMutation();
        fixture.saved()[2].gradientState().beforeMutation();
        rejects(fixture.node(), "A graph input was modified after forward");
        fixture.node().release();
    }

    @Test
    void backendAndReleasedGuardsPrecedeVersionChecks() {
        Fixture fixture = fixture(DType.FLOAT32, 3, true);
        fixture.inputs()[0].gradientState().beforeMutation();
        Tensor.setBackend(new CpuBackend());
        rejects(fixture.node(), "Backend changed between forward and backward");
        fixture.node().release();
        rejects(fixture.node(), "Backward graph was already released; run forward again");
    }

    private static void checkMutation(DType dtype, boolean aliases, int count, int index, boolean saved) {
        Fixture fixture = fixture(dtype, count, aliases);
        (saved ? fixture.saved() : fixture.inputs())[index].gradientState().beforeMutation();
        rejects(fixture.node(), saved ? "A tensor needed for backward was modified after forward"
                : "A graph input was modified after forward");
        fixture.node().release();
    }

    private static Fixture fixture(DType dtype, int count, boolean aliases) {
        Tensor[] inputs = tensors(dtype, count), saved = tensors(dtype, count);
        BackwardNode node = new BackwardNode(gradient -> new Tensor[count],
                aliases ? views(inputs) : inputs, aliases ? views(saved) : saved, true);
        return new Fixture(node, inputs, saved);
    }

    private static Tensor[] tensors(DType dtype, int count) {
        Tensor[] result = new Tensor[count];
        for (int index = 0; index < count; index++) { result[index] = new Tensor(dtype, 2, 3); }
        return result;
    }

    private static Tensor[] views(Tensor[] roots) {
        Tensor[] result = new Tensor[roots.length];
        for (int index = 0; index < result.length; index++) { result[index] = roots[index].view(1, 2, 3); }
        return result;
    }

    private static void rejects(BackwardNode node, String message) {
        assertEquals(message, assertThrows(IllegalStateException.class, node::validate).getMessage());
    }

    private record Fixture(BackwardNode node, Tensor[] inputs, Tensor[] saved) { }
}
