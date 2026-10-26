package io.github.kirstenali.deepj.persistence;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

class CheckpointInputTest {
    @TempDir Path directory;

    @Test
    void positionsExcludeBufferedReadAheadAndSkipsRemainExactAcrossBufferBoundaries() throws IOException {
        byte[] bytes = bytes();
        Path path = Files.write(directory.resolve("positions.bin"), bytes);
        try (var input = new CheckpointInput(path)) {
            assertEquals(ByteBuffer.wrap(bytes).getInt(), input.data().readInt());
            assertEquals(4, input.position());
            input.skip(80000);
            assertEquals(80004, input.position());
            assertEquals(bytes[80004] & 0xff, input.data().readUnsignedByte());
            assertEquals(80005, input.position());
            input.skip(input.size() - input.position());
            assertEquals(bytes.length, input.position());
            assertEquals(-1, input.data().read());
        }
    }

    @Test
    void rewindDiscardsReadAheadAndRestartsTheSameFileAtZero() throws IOException {
        byte[] bytes = bytes();
        Path path = Files.write(directory.resolve("rewind.bin"), bytes);
        try (var input = new CheckpointInput(path)) {
            input.skip(70001);
            input.data().readInt();
            input.rewind();
            assertEquals(0, input.position());
            assertEquals(ByteBuffer.wrap(bytes).getLong(), input.data().readLong());
            assertEquals(8, input.position());
            assertThrows(IOException.class, () -> input.requireRemaining(-1));
            assertThrows(IOException.class, () -> input.skip(bytes.length));
            assertEquals(8, input.position());
        }
    }

    @Test
    void closingTheOwnerAlsoClosesItsCurrentBufferedView() throws IOException {
        Path path = Files.write(directory.resolve("closed.bin"), new byte[128]);
        var input = new CheckpointInput(path);
        input.data().readInt();
        input.close();
        assertThrows(IOException.class, () -> input.data().readInt());
    }

    private static byte[] bytes() {
        byte[] result = new byte[3 * 65536 + 123];
        for (int index = 0; index < result.length; index++) {
            result[index] = (byte) (index * 17 + 23);
        }
        return result;
    }
}
