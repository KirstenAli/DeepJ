package io.github.kirstenali.deepj.distributed;

import java.io.IOException;
import java.util.Random;

/** Deterministic shuffled rank-strided partitions; no sample is padded or repeated within an epoch. */
public final class ShardSampler implements WorkerState {
    private final int samples;
    private final int rank;
    private final int workers;
    private final long seed;
    private long epoch;
    private int position;
    private int[] order;

    public ShardSampler(int samples, int rank, int workers, long seed) {
        if (samples < 0 || workers < 1 || rank < 0 || rank >= workers) {
            throw new IllegalArgumentException("Invalid data shard configuration");
        }
        this.samples = samples;
        this.rank = rank;
        this.workers = workers;
        this.seed = seed;
        shuffle();
    }

    public int samples() { return samples; }
    public long epoch() { return epoch; }
    public int remaining() { return localSamples() - position; }
    public int localSamples() { return samples <= rank ? 0 : 1 + (samples - 1 - rank) / workers; }

    public int stepsPerEpoch(int localBatchSize) {
        positive(localBatchSize);
        long largest = ((long) samples + workers - 1) / workers;
        return (int) ((largest + localBatchSize - 1) / localBatchSize);
    }

    public int[] next(int batchSize) {
        positive(batchSize);
        int[] indices = new int[Math.min(batchSize, remaining())];
        for (int index = 0; index < indices.length; index++) {
            indices[index] = order[(position + index) * workers + rank];
        }
        position += indices.length;
        return indices;
    }

    public void setEpoch(long epoch) {
        if (epoch < 0) throw new IllegalArgumentException("Epoch must be nonnegative");
        this.epoch = epoch;
        position = 0;
        shuffle();
    }

    private void shuffle() {
        order = java.util.stream.IntStream.range(0, samples).toArray();
        Random random = new Random(seed + epoch * 0x9E3779B97F4A7C15L);
        for (int index = samples - 1; index > 0; index--) {
            int other = random.nextInt(index + 1), saved = order[index];
            order[index] = order[other];
            order[other] = saved;
        }
    }

    @Override public byte[] snapshot() {
        return Wire.encode(output -> {
            output.writeInt(1);
            output.writeInt(samples);
            output.writeInt(rank);
            output.writeInt(workers);
            output.writeLong(seed);
            output.writeLong(epoch);
            output.writeInt(position);
        });
    }

    private Cursor read(byte[] bytes) {
        try (var input = Wire.input(bytes)) {
            validateHeader(input);
            Cursor cursor = new Cursor(input.readLong(), input.readInt());
            if (cursor.epoch() < 0 || cursor.position() < 0 || cursor.position() > localSamples()
                    || input.read() != -1) {
                throw new IllegalArgumentException("Invalid shard cursor");
            }
            return cursor;
        } catch (IOException failure) { throw new IllegalArgumentException("Invalid shard state", failure); }
    }

    private void validateHeader(java.io.DataInputStream input) throws IOException {
        if (input.readInt() != 1 || input.readInt() != samples || input.readInt() != rank
                || input.readInt() != workers || input.readLong() != seed) {
            throw new IllegalArgumentException("Checkpoint shard configuration differs");
        }
    }

    @Override public void validate(byte[] bytes) { read(bytes); }
    @Override public void restore(byte[] bytes) {
        Cursor cursor = read(bytes);
        setEpoch(cursor.epoch());
        position = cursor.position();
    }

    private static void positive(int value) {
        if (value < 1) throw new IllegalArgumentException("Batch size must be positive");
    }

    private record Cursor(long epoch, int position) {}
}
