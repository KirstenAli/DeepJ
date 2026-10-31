package io.github.kirstenali.deepj.persistence;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Path;

/** Buffered payload I/O with logical positions that exclude bytes read ahead by the buffer. */
final class CheckpointInput implements AutoCloseable {
    private final FileChannel channel;
    private PositionBuffer buffer;
    private DataInputStream data;

    CheckpointInput(Path path) throws IOException {
        channel = FileChannel.open(path);
        bindStream();
    }

    DataInputStream data() { return data; }
    long position() throws IOException { return channel.position() - buffer.unread(); }
    long size() throws IOException { return channel.size(); }

    void requireRemaining(long bytes) throws IOException {
        if (bytes < 0 || bytes > size() - position()) throw new IOException("Truncated checkpoint tensor");
    }

    void skip(long bytes) throws IOException {
        requireRemaining(bytes);
        data.skipNBytes(bytes);
    }

    void rewind() throws IOException {
        seek(0);
    }

    void seek(long position) throws IOException {
        if (position < 0 || position > size()) throw new IOException("Invalid checkpoint position");
        channel.position(position);
        // The discarded wrapper owns no separate file descriptor; this reader owns the shared channel.
        bindStream();
    }

    private void bindStream() {
        buffer = new PositionBuffer(Channels.newInputStream(channel));
        data = new DataInputStream(buffer);
    }

    @Override public void close() throws IOException { data.close(); }

    private static final class PositionBuffer extends BufferedInputStream {
        PositionBuffer(InputStream input) { super(input, 64 * 1024); }
        int unread() { return count - pos; }
    }
}
