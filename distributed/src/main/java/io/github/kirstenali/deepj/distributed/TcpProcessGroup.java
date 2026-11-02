package io.github.kirstenali.deepj.distributed;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/** TCP reduction/broadcast with rank-zero coordination and an absolute deadline for each collective. */
public final class TcpProcessGroup implements ProcessGroup {
    private final TcpSettings settings;
    private final TcpExchange transport;
    private final InetSocketAddress address;
    private final ScheduledExecutorService deadlines;
    private volatile String failure;
    private long sequence;

    TcpProcessGroup(TcpSettings settings, TcpExchange transport, InetSocketAddress address) {
        this.settings = settings;
        this.transport = transport;
        this.address = address;
        deadlines = Executors.newSingleThreadScheduledExecutor(task -> {
            Thread thread = new Thread(task, "deepj-distributed-deadline");
            thread.setDaemon(true);
            return thread;
        });
    }

    public static TcpProcessGroup coordinator(InetSocketAddress address, int workers,
                                              String session, Duration timeout) throws IOException {
        TcpSettings settings = TcpSettings.of(0, workers, session, timeout);
        var coordinator = new TcpCoordinator(Objects.requireNonNull(address), settings);
        return new TcpProcessGroup(settings, coordinator, coordinator.address());
    }

    public static TcpProcessGroup connect(InetSocketAddress address, int rank, int workers,
                                          String session, Duration timeout) throws IOException {
        TcpSettings settings = TcpSettings.of(rank, workers, session, timeout);
        if (rank == 0) throw new IllegalArgumentException("Rank zero must create the coordinator");
        return new TcpProcessGroup(settings, new TcpWorker(Objects.requireNonNull(address), settings), address);
    }

    public InetSocketAddress address() { return address; }
    @Override public int rank() { return settings.rank(); }
    @Override public int worldSize() { return settings.worldSize(); }

    @Override public void sum(String key, float[] values) {
        CollectiveValues.copy(exchange(Wire.SUM, key, CollectiveValues.floats(values)), values);
    }

    @Override public void broadcast(String key, float[] values) {
        CollectiveValues.copy(exchange(Wire.BROADCAST, key, CollectiveValues.floats(values)), values);
    }

    @Override public List<byte[]> gather(String key, byte[] value) {
        try { return CollectiveValues.unpack(exchange(Wire.GATHER, key, value), worldSize()); }
        catch (IOException error) {
            abort(error.getMessage());
            throw failed(error);
        }
    }

    private synchronized byte[] exchange(int operation, String key, byte[] bytes) {
        Objects.requireNonNull(key, "key");
        if (key.length() > 4096 || failure != null) throw failed(null);
        var deadline = deadlines.schedule(() -> abort("Collective timed out"),
                settings.timeoutMillis(), TimeUnit.MILLISECONDS);
        try {
            byte[] result = transport.exchange(sequence++, operation, key, bytes);
            if (failure != null) throw failed(null);
            return result;
        } catch (IOException | RuntimeException error) {
            abort(error.getMessage());
            throw failed(error);
        }
        finally { deadline.cancel(false); }
    }

    private CollectiveException failed(Throwable cause) {
        return new CollectiveException("Rank " + rank() + ": " + (failure == null ? "Invalid collective" : failure),
                cause);
    }

    @Override public void abort(String reason) {
        failure = Objects.requireNonNullElse(reason, "Collective failed");
        transport.close();
        deadlines.shutdownNow();
    }

    @Override public void close() { abort("Process group closed"); }
}
