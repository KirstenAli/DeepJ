package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.net.InetSocketAddress;
import java.time.Duration;

public final class ForkWorker {
    private ForkWorker() {}

    public static void main(String[] args) throws Exception {
        configure(args[6]);
        int rank = Integer.parseInt(args[2]), workers = Integer.parseInt(args[3]);
        try (ProcessGroup group = group(args, rank, workers)) {
            if (java.util.Set.of("probe", "crash", "stall", "bad-key", "bad-size").contains(args[7])) {
                probe(group, args[7]);
                return;
            }
            train(args, rank, workers, group);
        } finally { Tensor.backend().releaseResources(); }
    }

    private static ProcessGroup group(String[] args, int rank, int workers) throws Exception {
        Duration timeout = Duration.ofSeconds(20);
        if (args[8].equals("pipe")) return PipeTransport.worker(rank, workers, timeout);
        return TcpProcessGroup.connect(new InetSocketAddress("127.0.0.1", Integer.parseInt(args[8])),
                rank, workers, "test", timeout);
    }

    private static void configure(String backend) {
        if (backend.equals("metal")) {
            if (!MetalBackend.isAvailable()) throw new IllegalStateException("A physical Metal device is required");
            Tensor.setBackend(new MetalBackend());
        } else Tensor.setBackend(new CpuBackend());
    }

    private static void train(String[] args, int rank, int workers, ProcessGroup group) throws Exception {
        String family = args[7].equals("wrong-layout") ? "gru" : args[0];
        TrainingFixture fixture = WorkerFault.inject(TrainingFixture.create(family,
                DType.valueOf(args[1]), 999 + rank, rank, workers), args[7]);
        var options = new DistributedOptions(7, args[7].equals("buffers-local")
                ? BufferPolicy.LOCAL : BufferPolicy.BROADCAST_FROM_ROOT);
        try (var trainer = new DistributedTrainer(group, fixture.objective(), fixture.optimizer(), options)) {
            runSteps(args, group, trainer, fixture);
            if (args[7].startsWith("checkpoint")) resume(args, group, trainer, fixture);
        }
    }

    private static void runSteps(String[] args, ProcessGroup group, DistributedTrainer trainer,
                                 TrainingFixture fixture) {
        for (int step = 0; step < Integer.parseInt(args[5]); step++) {
            float loss = trainer.trainStep(Integer.parseInt(args[4]));
            assertBackend(fixture, args[6]);
            group.gather("test.report", TrainingReport.encode(fixture, loss));
            if (args[7].startsWith("checkpoint") && step == 0) {
                DistributedCheckpoint.save(trainer, java.nio.file.Path.of(args[9]));
            }
        }
    }

    private static void resume(String[] args, ProcessGroup group, DistributedTrainer trainer, TrainingFixture fixture)
            throws Exception {
        if (args[7].equals("checkpoint-mixed")) {
            CheckpointMixingTest.relabel(java.nio.file.Path.of(args[9]), group.rank());
            group.barrier("test.mixed.ready");
        }
        DistributedCheckpoint.load(trainer, java.nio.file.Path.of(args[9]));
        float loss = trainer.trainStep(Integer.parseInt(args[4]));
        group.gather("test.report", TrainingReport.encode(fixture, loss));
    }

    static void assertBackend(TrainingFixture fixture, String backend) {
        if (!backend.equals("metal")) return;
        for (var parameter : fixture.objective().model().parameters()) {
            if (parameter.grad().getGpuTag() == null) throw new AssertionError("Gradient left Metal");
        }
    }

    private static void probe(ProcessGroup group, String mode) throws Exception {
        if (mode.equals("crash")) System.exit(3);
        if (mode.equals("stall")) Thread.sleep(10000);
        String key = mode.equals("bad-key") ? "wrong" : "probe";
        float[] values = mode.equals("bad-size") ? new float[3] : new float[]{group.rank(), 2};
        group.sum(key, values);
        group.broadcast("broadcast", values);
        group.gather("probe.report", CollectiveValues.floats(values));
    }
}
