package io.github.kirstenali.deepj.distributed;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;

final class TcpPeer implements AutoCloseable {
    final DataInputStream input;
    final DataOutputStream output;
    private final Socket socket;

    TcpPeer(Socket socket, int timeoutMillis) throws IOException {
        this.socket = socket;
        try {
            socket.setSoTimeout(timeoutMillis);
            socket.setTcpNoDelay(true);
            input = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
            output = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
        } catch (IOException failure) {
            socket.close();
            throw failure;
        }
    }

    TcpPeer(InputStream input, OutputStream output) {
        socket = null;
        this.input = new DataInputStream(new BufferedInputStream(input));
        this.output = new DataOutputStream(new BufferedOutputStream(output));
    }

    void hello(int rank, int world, String session) throws IOException {
        output.writeInt(Wire.MAGIC);
        output.writeInt(Wire.VERSION);
        output.writeInt(rank);
        output.writeInt(world);
        output.writeUTF(session);
        output.flush();
    }

    void request(long sequence, int operation, String key, byte[] bytes) throws IOException {
        output.writeLong(sequence);
        output.writeInt(operation);
        output.writeUTF(key);
        Wire.writeBytes(output, bytes);
        output.flush();
    }

    byte[] receive(long sequence, int operation, String key) throws IOException {
        if (input.readLong() != sequence || input.readInt() != operation || !input.readUTF().equals(key)) {
            throw new IOException("Collective order or key differs between workers");
        }
        return Wire.readBytes(input);
    }

    void reply(byte[] bytes) throws IOException {
        Wire.writeBytes(output, bytes);
        output.flush();
    }
    byte[] response() throws IOException { return Wire.readBytes(input); }

    @Override public void close() {
        try {
            if (socket != null) socket.close();
            else closeStreams();
        }
        catch (IOException ignored) { /* Closing a failed connection is best effort. */ }
    }

    private void closeStreams() throws IOException {
        try { input.close(); }
        finally { output.close(); }
    }
}
