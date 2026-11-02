package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import static org.junit.jupiter.api.Assertions.*;

/** An explicitly requested GPU check fails, rather than falling back, when Metal is unavailable. */
@Timeout(120)
@EnabledIfSystemProperty(named = "deepj.distributed.requireMetal", matches = "true")
class DistributedMetalTest {
    @BeforeEach void metal() {
        assertTrue(MetalBackend.isAvailable(), "A physical Metal device is required for this qualification run");
        Tensor.setBackend(new MetalBackend());
    }

    @AfterEach void release() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(new CpuBackend());
    }

    @ParameterizedTest @MethodSource("io.github.kirstenali.deepj.distributed.DistributedParityTest#families")
    void physicalMetalWorkersMatchExistingTraining(String family, DType dtype) throws Exception {
        DistributedParityTest.run(family, dtype, "metal");
    }

    @ParameterizedTest @EnumSource(DType.class)
    void physicalMetalWorkersResumeTheirCheckpoints(DType dtype) throws Exception {
        CheckpointTest.clusterCheckpoint(dtype, "metal");
    }
}
