package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import static org.junit.jupiter.api.Assertions.*;

@Timeout(60)
class DistributedParityTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    static Stream<Arguments> families() {
        return Stream.concat(ModelCases.tensorFamilies().stream(), DecoderCases.families().stream())
                .flatMap(family -> Arrays.stream(DType.values()).map(dtype -> Arguments.of(family, dtype)));
    }

    @ParameterizedTest(name = "{0} {1}") @MethodSource("families")
    void subprocessWorkersMatchTheExistingSingleWorkerTrainingAndClipping(String family, DType dtype) throws Exception {
        run(family, dtype, "cpu");
    }

    static void run(String family, DType dtype, String backend) throws Exception {
        int[] batches = {2, 1};
        try (var cluster = new ForkCluster(family, dtype, batches, 3, backend, "train")) {
            TrainingFixture actual = TrainingFixture.create(family, dtype, 123, 0, 2);
            TrainingFixture expected = TrainingFixture.create(family, dtype, 123, 0, 1);
            var trainer = new DistributedTrainer(cluster.group(), actual.objective(), actual.optimizer(),
                    new DistributedOptions(7, BufferPolicy.BROADCAST_FROM_ROOT));
            ShardSampler[] samplers = samplers(2);
            for (int step = 0; step < 3; step++) {
                int[] indices = indices(samplers, batches);
                float loss = trainer.trainStep(batches[0]);
                ForkWorker.assertBackend(actual, backend);
                compare(cluster, actual, expected, family, dtype, indices, loss);
                assertEquals(indices.length, trainer.lastStep().samples());
            }
            cluster.successful();
        }
    }

    static ShardSampler[] samplers(int workers) {
        return IntStream.range(0, workers).mapToObj(rank -> new ShardSampler(7, rank, workers, 42))
                .toArray(ShardSampler[]::new);
    }

    static int[] indices(ShardSampler[] samplers, int[] batches) {
        var values = IntStream.builder();
        for (int rank = 0; rank < samplers.length; rank++) {
            for (int index : samplers[rank].next(batches[rank])) { values.add(index); }
        }
        return values.build().toArray();
    }

    static void compare(ForkCluster cluster, TrainingFixture actual, TrainingFixture expected, String family,
                        DType dtype, int[] indices, float loss) throws Exception {
        byte[] local = TrainingReport.encode(actual, loss);
        List<byte[]> reports = cluster.group().gather("test.report", local);
        for (byte[] report : reports) { assertArrayEquals(local, report, "Replicas and AdamW state must agree"); }
        float referenceLoss = expected.referenceStep(family, indices);
        TrainingReport.compare(TrainingReport.encode(expected, referenceLoss), local, actual, tolerance(dtype));
    }

    private static float tolerance(DType dtype) {
        boolean metal = Tensor.backend() instanceof io.github.kirstenali.deepj.tensor.metal.MetalBackend;
        if (dtype == DType.FLOAT32) return metal ? 0.0002f : 0.00002f;
        if (dtype == DType.FLOAT16) return metal ? 0.003f : 0.002f;
        return 0.02f;
    }
}
