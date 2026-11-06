package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.onnx.proto.Onnx;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.stream.Collectors;

final class ExternalTensorData {
    private ExternalTensorData() {}

    static ByteBuffer read(Onnx.TensorProto tensor, Path directory, long length)
            throws IOException {
        Map<String, String> data =
                tensor.getExternalDataList().stream()
                        .collect(
                                Collectors.toMap(
                                        Onnx.StringStringEntryProto::getKey,
                                        Onnx.StringStringEntryProto::getValue));
        Path file = externalPath(directory, data.get("location"));
        long offset = Long.parseLong(data.getOrDefault("offset", "0"));
        long declared = Long.parseLong(data.getOrDefault("length", Long.toString(length)));
        if (offset < 0 || declared != length)
            throw new IOException("Invalid ONNX external tensor range");
        return readRange(file, offset, length);
    }

    private static Path externalPath(Path directory, String location) throws IOException {
        if (location == null || Path.of(location).isAbsolute())
            throw new IOException("Invalid external data location");
        for (Path component : Path.of(location)) {
            if (component.toString().equals(".."))
                throw new IOException("External ONNX data cannot use parent paths");
        }
        Path base = directory.toRealPath(), file = base.resolve(location).normalize().toRealPath();
        if (!file.startsWith(base))
            throw new IOException("External ONNX data escapes the model directory");
        return file;
    }

    private static ByteBuffer readRange(Path file, long offset, long length) throws IOException {
        try (var channel = FileChannel.open(file, StandardOpenOption.READ)) {
            if (Math.addExact(offset, length) > channel.size())
                throw new IOException("Truncated ONNX external tensor");
            ByteBuffer buffer =
                    ByteBuffer.allocate(Math.toIntExact(length)).order(ByteOrder.LITTLE_ENDIAN);
            channel.position(offset);
            while (buffer.hasRemaining()) {
                if (channel.read(buffer) < 0)
                    throw new IOException("Truncated ONNX external tensor");
            }
            return buffer.flip();
        }
    }
}
