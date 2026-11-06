package io.github.kirstenali.deepj.onnx;

import static org.junit.jupiter.api.Assertions.*;

import io.github.kirstenali.deepj.onnx.proto.Onnx;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.api.io.TempDir;

class LoaderValidationTest {
    @TempDir Path directory;

    @TestFactory
    Stream<DynamicTest> rejectsUnsupportedOrMalformedGraphsBeforeExecution() {
        return List.of("Mystery", "QuantizeLinear", "SequenceConstruct").stream()
                .map(
                        operator ->
                                DynamicTest.dynamicTest(
                                        operator,
                                        () -> {
                                            Path path =
                                                    TestModels.write(
                                                            directory,
                                                            TestModels.graph(
                                                                    TestModels.unary(operator),
                                                                    1,
                                                                    2));
                                            assertFalse(OnnxModel.inspect(path).supported());
                                            assertThrows(
                                                    IOException.class, () -> OnnxModel.load(path));
                                        }));
    }

    @Test
    void rejectsFutureOpsetsAndUnknownAttributes() throws Exception {
        var model = TestModels.graph(TestModels.unary("Relu"), 1, 2).toBuilder();
        model.setOpsetImport(0, model.getOpsetImport(0).toBuilder().setVersion(999));
        assertRejected(model.build(), "opset");
        model.setOpsetImport(0, model.getOpsetImport(0).toBuilder().setVersion(18));
        model.getGraphBuilder().getNodeBuilder(0).addAttribute(TestModels.integer("made_up", 1));
        assertRejected(model.build(), "attribute");
    }

    @Test
    void rejectsMissingRequiredInputsAndDuplicateValues() throws Exception {
        var model = TestModels.graph(TestModels.unary("Add"), 1, 2).toBuilder();
        assertRejected(model.build(), "count");
        model.getGraphBuilder().getNodeBuilder(0).addInput("missing");
        assertRejected(model.build(), "Undefined");
        model.getGraphBuilder().getNodeBuilder(0).setInput(1, "X").setOutput(0, "X");
        assertRejected(model.build(), "duplicate");
    }

    @Test
    void rejectsWrongAttributeTypesAndUnsupportedModes() throws Exception {
        var node =
                TestModels.unary("Transpose").toBuilder()
                        .addAttribute(TestModels.integer("perm", 0));
        assertRejected(TestModels.graph(node.build(), 1, 2), "attribute type");
        node =
                TestModels.unary("Resize").toBuilder()
                        .addInput("")
                        .addInput("S")
                        .addAttribute(TestModels.integer("antialias", 1));
        assertRejected(TestModels.graph(node.build(), 1, 2), "antialias");
    }

    @Test
    void rejectsUnsupportedTensorTypesAndBrokenWireData() throws Exception {
        assertRejected(TestModels.graph(TestModels.unary("Identity"), 11, 2), "element type");
        Path broken = directory.resolve("broken.onnx");
        Files.write(broken, new byte[] {10, 127});
        assertThrows(IOException.class, () -> OnnxModel.load(broken));
    }

    @Test
    void enforcesImportBudgets() throws Exception {
        Path path = TestModels.write(directory, TestModels.fixture("mlp"));
        assertThrows(
                IOException.class,
                () -> OnnxModel.load(path, OnnxRegistry.standard(), new OnnxOptions(1, 1024, 4)));
        assertThrows(
                IOException.class,
                () -> OnnxModel.load(path, OnnxRegistry.standard(), new OnnxOptions(4096, 1, 4)));
    }

    @Test
    void customDomainsRequireExplicitRegistration() throws Exception {
        var node = TestModels.unary("Twice").toBuilder().setDomain("example.deepj");
        var model =
                TestModels.graph(node.build(), 1, 2).toBuilder()
                        .addOpsetImport(TestModels.opset("example.deepj", 1))
                        .build();
        Path path = TestModels.write(directory, model);
        assertFalse(OnnxModel.inspect(path).supported());
        var registry = twiceRegistry();
        try (var imported = OnnxModel.load(path, registry, OnnxOptions.defaults())) {
            assertArrayEquals(
                    new float[] {2, 4},
                    imported.forward(Tensor.from(new float[] {1, 2}, 2)).toFloatArray());
        }
    }

    @Test
    void executionValidatesTypesShapesNamesAndClosedModels() throws Exception {
        Path path =
                TestModels.write(directory, TestModels.graph(TestModels.unary("Identity"), 1, 2));
        var model = OnnxModel.load(path);
        try (model) {
            assertThrows(IllegalArgumentException.class, () -> model.run(Map.of()));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> model.run(Map.of("other", OnnxValue.of(Tensor.ones(DType.FLOAT32, 2)))));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> model.forward(Tensor.ones(DType.FLOAT32, 3)));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> model.run(Map.of("X", OnnxValue.integers(new long[] {1, 2}, 2))));
        }
        assertThrows(
                IllegalStateException.class, () -> model.forward(Tensor.ones(DType.FLOAT32, 2)));
    }

    @Test
    void loopsStopAtTheConfiguredBound() throws Exception {
        Path path = TestModels.write(directory, TestModels.fixture("loop_3"));
        try (var model =
                OnnxModel.load(path, OnnxRegistry.standard(), new OnnxOptions(4096, 4096, 2))) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> model.run(Map.of("X", OnnxValue.of(Tensor.ones(DType.FLOAT32, 2)))));
        }
    }

    private void assertRejected(Onnx.ModelProto model, String message) throws IOException {
        Path path = TestModels.write(directory, model);
        assertTrue(OnnxModel.inspect(path).problems().stream().anyMatch(p -> p.contains(message)));
        assertThrows(IOException.class, () -> OnnxModel.load(path));
    }

    private static OnnxRegistry twiceRegistry() {
        return OnnxRegistry.standard()
                .register(
                        "example.deepj",
                        "Twice",
                        1,
                        1,
                        Set.of(),
                        c -> List.of(OnnxValue.of(c.tensor(0).multiplyScalar(2))));
    }
}
