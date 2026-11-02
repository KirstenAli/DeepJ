package io.github.kirstenali.deepj.distributed;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

final class CoordinatorProtocol implements TcpExchange {
    private final TcpSettings settings;
    private final TcpPeer[] ranks;
    private final List<TcpPeer> connections = new CopyOnWriteArrayList<>();

    CoordinatorProtocol(TcpSettings settings) {
        this.settings = settings;
        ranks = new TcpPeer[settings.worldSize()];
    }

    int connectedWorkers() { return connections.size(); }

    void register(TcpPeer peer) throws IOException {
        connections.add(peer);
        int rank = handshake(peer);
        if (ranks[rank] != null) throw new IOException("Duplicate worker rank");
        ranks[rank] = peer;
    }

    @Override public byte[] exchange(long sequence, int operation, String key, byte[] bytes) throws IOException {
        List<byte[]> values = collect(sequence, operation, key, bytes);
        byte[] result = result(operation, values);
        for (int rank = 1; rank < ranks.length; rank++) { ranks[rank].reply(result); }
        return result;
    }

    private int handshake(TcpPeer peer) throws IOException {
        if (peer.input.readInt() != Wire.MAGIC || peer.input.readInt() != Wire.VERSION) {
            throw new IOException("Incompatible distributed protocol");
        }
        int rank = peer.input.readInt(), world = peer.input.readInt();
        String session = peer.input.readUTF();
        if (rank < 1 || rank >= ranks.length || world != ranks.length || !session.equals(settings.session())) {
            throw new IOException("Worker session, rank, or count differs");
        }
        return rank;
    }

    private List<byte[]> collect(long sequence, int operation, String key, byte[] bytes) throws IOException {
        List<byte[]> values = new ArrayList<>();
        values.add(bytes);
        for (int rank = 1; rank < ranks.length; rank++) {
            values.add(ranks[rank].receive(sequence, operation, key));
        }
        return values;
    }

    private static byte[] result(int operation, List<byte[]> values) throws IOException {
        return switch (operation) {
            case Wire.SUM -> CollectiveValues.sum(values);
            case Wire.BROADCAST -> broadcast(values);
            case Wire.GATHER -> CollectiveValues.gathered(values);
            default -> throw new IOException("Unknown collective operation");
        };
    }

    private static byte[] broadcast(List<byte[]> values) throws IOException {
        int length = values.get(0).length;
        for (byte[] value : values) {
            if (value.length != length) throw new IOException("Broadcast sizes differ");
        }
        return values.get(0);
    }

    @Override public void close() {
        for (TcpPeer peer : connections) { peer.close(); }
    }
}
