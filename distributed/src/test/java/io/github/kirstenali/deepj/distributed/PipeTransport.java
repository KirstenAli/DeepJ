package io.github.kirstenali.deepj.distributed;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/** Exercises the production framing, coordinator, reductions and deadlines without requiring a listening socket. */
final class PipeTransport implements TcpExchange {
    private final CoordinatorProtocol protocol;
    private final List<TcpPeer> peers;
    private boolean connected;

    PipeTransport(List<Process> children, TcpSettings settings) {
        protocol = new CoordinatorProtocol(settings);
        peers = new ArrayList<>();
        for (Process process : children) {
            peers.add(new TcpPeer(process.getInputStream(), process.getOutputStream()));
        }
    }

    @Override public byte[] exchange(long sequence, int operation, String key, byte[] bytes) throws IOException {
        if (!connected) {
            for (TcpPeer peer : peers) { protocol.register(peer); }
            connected = true;
        }
        return protocol.exchange(sequence, operation, key, bytes);
    }

    @Override public void close() {
        for (TcpPeer peer : peers) { peer.close(); }
    }

    static ProcessGroup worker(int rank, int workers, Duration timeout) throws IOException {
        TcpSettings settings = TcpSettings.of(rank, workers, "test", timeout);
        TcpPeer peer = new TcpPeer(System.in, System.out);
        peer.hello(rank, workers, "test");
        return new TcpProcessGroup(settings, participant(peer), new InetSocketAddress(0));
    }

    private static TcpExchange participant(TcpPeer peer) {
        return new TcpExchange() {
            @Override public byte[] exchange(long sequence, int operation, String key, byte[] bytes)
                    throws IOException {
                peer.request(sequence, operation, key, bytes);
                return peer.response();
            }
            @Override public void close() { peer.close(); }
        };
    }
}
