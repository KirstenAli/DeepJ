package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.onnx.proto.Onnx;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

final class TestModels {
    private TestModels() {}

    static Onnx.ModelProto fixture(String name) throws IOException {
        return Onnx.ModelProto.parseFrom(Files.readAllBytes(ReferenceFixturesTest.FIXTURES.resolve(name + ".onnx")));
    }

    static Path write(Path directory, Onnx.ModelProto model) throws IOException {
        Path path = Files.createTempFile(directory, "graph-", ".onnx");
        Files.write(path, model.toByteArray());
        return path;
    }

    static Onnx.ModelProto graph(Onnx.NodeProto node, int type, long... dimensions) {
        var graph = Onnx.GraphProto.newBuilder().setName("test").addNode(node).addInput(value("X", type, dimensions))
                        .addOutput(value("Y", type, dimensions));
        return Onnx.ModelProto.newBuilder().setIrVersion(10).setGraph(graph)
                .addOpsetImport(Onnx.OperatorSetIdProto.newBuilder().setDomain("").setVersion(18)).build();
    }

    static Onnx.ValueInfoProto value(String name, int type, long... dimensions) {
        var shape = Onnx.TensorShapeProto.newBuilder();
        for (long dimension : dimensions) {
            shape.addDim(Onnx.TensorShapeProto.Dimension.newBuilder().setDimValue(dimension));
        }
        var tensor = Onnx.TypeProto.Tensor.newBuilder().setElemType(type).setShape(shape);
        return Onnx.ValueInfoProto.newBuilder().setName(name).setType(Onnx.TypeProto.newBuilder().setTensorType(tensor))
                .build();
    }

    static Onnx.NodeProto unary(String operator) {
        return Onnx.NodeProto.newBuilder().setOpType(operator).addInput("X").addOutput("Y").build();
    }

    static Onnx.AttributeProto integer(String name, long value) {
        return Onnx.AttributeProto.newBuilder().setName(name).setType(Onnx.AttributeProto.AttributeType.INT).setI(value)
                .build();
    }
}
