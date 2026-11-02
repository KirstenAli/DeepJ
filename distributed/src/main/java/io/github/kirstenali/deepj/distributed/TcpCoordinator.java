package io.github.kirstenali.deepj.distributed;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;

final class TcpCoordinator implements TcpExchange {
    private final TcpSettings settings;
    private final ServerSocket server;
    private final CoordinatorProtocol protocol;

    TcpCoordinator(InetSocketAddress address, TcpSettings settings) throws IOException {
        this.settings = settings;
        protocol = new CoordinatorProtocol(settings);
        server = new ServerSocket();
        try {
            server.bind(address);
            server.setSoTimeout(settings.timeoutMillis());
        } catch (IOException failure) {
            server.close();
            throw failure;
        }
    }

    InetSocketAddress address() { return (InetSocketAddress) server.getLocalSocketAddress(); }

    @Override public byte[] exchange(long sequence, int operation, String key, byte[] bytes) throws IOException {
        while (protocol.connectedWorkers() < settings.worldSize() - 1) {
            protocol.register(new TcpPeer(server.accept(), settings.timeoutMillis()));
        }
        server.close();
        return protocol.exchange(sequence, operation, key, bytes);
    }

    @Override public void close() {
        try { server.close(); }
        catch (IOException ignored) { /* Close connected workers after a listener failure too. */ }
        protocol.close();
    }
}
