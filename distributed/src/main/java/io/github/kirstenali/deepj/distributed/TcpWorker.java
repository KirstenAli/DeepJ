package io.github.kirstenali.deepj.distributed;

import java.io.IOException;
import java.net.ConnectException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.concurrent.TimeUnit;

final class TcpWorker implements TcpExchange {
    private final TcpPeer peer;

    TcpWorker(InetSocketAddress address, TcpSettings settings) throws IOException {
        peer = new TcpPeer(connect(address, settings.timeoutMillis()), settings.timeoutMillis());
        try { peer.hello(settings.rank(), settings.worldSize(), settings.session()); }
        catch (IOException failure) {
            peer.close();
            throw failure;
        }
    }

    private static Socket connect(InetSocketAddress address, int timeoutMillis) throws IOException {
        long end = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMillis);
        while (true) {
            int remaining = (int) Math.max(1, TimeUnit.NANOSECONDS.toMillis(end - System.nanoTime()));
            try { return attempt(address, remaining); }
            catch (ConnectException failure) {
                if (System.nanoTime() >= end) throw failure;
                pause();
            }
        }
    }

    private static Socket attempt(InetSocketAddress address, int timeout) throws IOException {
        Socket socket = new Socket();
        try {
            socket.connect(address, timeout);
            return socket;
        } catch (IOException failure) {
            socket.close();
            throw failure;
        }
    }

    private static void pause() throws IOException {
        try { Thread.sleep(10); }
        catch (InterruptedException failure) {
            Thread.currentThread().interrupt();
            throw new IOException("Worker connection interrupted", failure);
        }
    }

    @Override public byte[] exchange(long sequence, int operation, String key, byte[] bytes) throws IOException {
        peer.request(sequence, operation, key, bytes);
        return peer.response();
    }

    @Override public void close() { peer.close(); }
}
