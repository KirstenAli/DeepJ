package io.github.kirstenali.deepj.onnx;

import static org.junit.jupiter.api.Assertions.*;

import io.github.kirstenali.deepj.onnx.proto.Onnx;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Map;

class DynamicGraphTest {
    @TempDir Path directory;

    @Test
    void symbolicBatchSizesVaryBetweenInvocations() throws Exception {
        var model = TestModels.fixture("mlp").toBuilder();
        symbolic(model.getGraphBuilder().getInputBuilder(0), "batch");
        symbolic(model.getGraphBuilder().getOutputBuilder(0), "batch");
        try (var imported = OnnxModel.load(TestModels.write(directory, model.build()))) {
            assertArrayEquals(new int[] {1, 4}, imported.forward(Tensor.ones(1, 3)).shape());
            assertArrayEquals(new int[] {5, 4}, imported.forward(Tensor.ones(5, 3)).shape());
            assertThrows(IllegalArgumentException.class, () -> imported.forward(Tensor.ones(5, 2)));
        }
    }

    @Test
    void sharedSymbolsMustAgreeEvenWhenBroadcastingWouldBePossible() throws Exception {
        var node = TestModels.unary("Add").toBuilder().addInput("B").build();
        var model = TestModels.graph(node, 1, 2, 3).toBuilder();
        model.getGraphBuilder().addInput(TestModels.value("B", 1, 2, 3));
        symbolic(model.getGraphBuilder().getInputBuilder(0), "batch");
        symbolic(model.getGraphBuilder().getInputBuilder(1), "batch");
        try (var imported = OnnxModel.load(TestModels.write(directory, model.build()))) {
            var inputs = Map.of("X", OnnxValue.of(Tensor.ones(2, 3)), "B", OnnxValue.of(Tensor.ones(1, 3)));
            assertThrows(IllegalArgumentException.class, () -> imported.run(inputs));
        }
    }

    @Test
    void aLiveModelCannotSilentlySwitchBackends() throws Exception {
        var original = Tensor.backend();
        try (var imported = OnnxModel.load(ReferenceFixturesTest.FIXTURES.resolve("mlp.onnx"))) {
            Tensor.setBackend(new CpuBackend());
            assertThrows(IllegalStateException.class, () -> imported.forward(Tensor.ones(2, 3)));
        } finally {
            Tensor.setBackend(original);
        }
    }

    @Test
    void functionOutputsCanAliasFormalInputs() throws Exception {
        var function = Onnx.FunctionProto.newBuilder().setDomain("example.deepj").setName("Alias").addInput("X")
                        .addOutput("X").addOpsetImport(Onnx.OperatorSetIdProto.newBuilder().setVersion(18));
        var node = TestModels.unary("Alias").toBuilder().setDomain("example.deepj").build();
        var model = TestModels.graph(node, 1, 2).toBuilder().addFunctions(function).addOpsetImport(
                                Onnx.OperatorSetIdProto.newBuilder().setDomain("example.deepj").setVersion(1));
        try (var imported = OnnxModel.load(TestModels.write(directory, model.build()))) {
            assertArrayEquals(new float[] {2, 3}, imported.forward(Tensor.from(new float[] {2, 3}, 2)).toFloatArray());
        }
    }

    private static void symbolic(Onnx.ValueInfoProto.Builder value, String name) {
        value.getTypeBuilder().getTensorTypeBuilder().getShapeBuilder().getDimBuilder(0).clearDimValue()
                .setDimParam(name);
    }
}
