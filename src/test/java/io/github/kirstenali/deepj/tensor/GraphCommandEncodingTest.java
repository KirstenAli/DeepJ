package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GraphCommandEncodingTest extends ComputeGraphTestSupport {

    private GpuBuffer a;
    private GpuBuffer b;
    private GpuBuffer c;
    private GpuBuffer d;
    private GpuBuffer e;

    @BeforeEach
    void allocateBuffers() {
        a = graph.newOutputBuffer(2, 3);
        b = graph.newOutputBuffer(2, 3);
        c = graph.newOutputBuffer(2, 3);
        d = graph.newOutputBuffer(2, 3);
        e = graph.newOutputBuffer(2, 3);
    }

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void batchedMatmulPreservesTransposeFlagsAndDimensions(boolean left, boolean right) {
        graph.recordBatchedMatmul(a, b, c, 2, 3, 4, 5, 6, left, right);
        assertCommand(ComputeGraph.OP_BATCHED_MATMUL, a.id, b.id, c.id, 2, 3, 4, 5, 6,
                left ? 1 : 0, right ? 1 : 0);
    }

    @Test
    void attentionCommandsPreserveTheirNativeFieldOrder() {
        graph.recordCausalMask(a, b, 2, 3);
        assertCommand(ComputeGraph.OP_CAUSAL_MASK, a.id, b.id, 2, 3);
        graph.recordCausalSoftmax(a, b, 2, 3, 0.5f);
        assertCommand(ComputeGraph.OP_CAUSAL_SOFTMAX, a.id, b.id, 2, 3, 3, bits(0.5f));
        graph.recordGatherRows(a, b, c, 2, 3, 4);
        assertCommand(ComputeGraph.OP_GATHER_ROWS, a.id, b.id, c.id, 2, 3, 4);
    }

    @Test
    void scaledSoftmaxBackwardPreservesScaleBitsAndShape() {
        graph.recordScaledSoftmaxBackward(a, b, c, -0.25f);
        assertCommand(ComputeGraph.OP_SCALED_SOFTMAX_BACKWARD,
                a.id, b.id, c.id, 2, 3, bits(-0.25f));
    }

    @Test
    void tiledAttentionRecordsItsLinearNormalizationState() {
        graph.recordCausalAttention(a, b, c, d, e, 2, 0.125f);
        assertCommand(ComputeGraph.OP_TILED_CAUSAL_ATTENTION,
                a.id, b.id, c.id, d.id, e.id, 2, 2, 2, bits(0.125f));
    }

    @Test
    void tiledAttentionBackwardRecordsEveryInputAndGradientBuffer() {
        GpuBuffer f = graph.newOutputBuffer(2, 1), g = graph.newOutputBuffer(2, 1);
        GpuBuffer h = graph.newOutputBuffer(2, 3), i = graph.newOutputBuffer(2, 3);
        GpuBuffer j = graph.newOutputBuffer(2, 3);
        graph.recordCausalAttentionBackward(a, b, c, d, e, f, g, h, i, j, 2, 0.125f);
        assertCommand(ComputeGraph.OP_TILED_CAUSAL_ATTENTION_BACKWARD,
                a.id, b.id, c.id, d.id, e.id, f.id, g.id, h.id, i.id, j.id, 2, 2, 2, bits(0.125f));
    }

    @Test
    void groupedAttentionRecordsDistinctQueryAndKeyValueHeadCounts() {
        graph.recordCausalAttention(a, b, c, d, e, 9, 3, 0.125f);
        assertCommand(ComputeGraph.OP_TILED_CAUSAL_ATTENTION,
                a.id, b.id, c.id, d.id, e.id, 2, 9, 3, bits(0.125f));
    }

    @Test
    void headCacheCommandsPreserveOffsetsAndPrefixDimensions() {
        graph.recordHeadCacheWrite(a, b, 2, 3);
        assertCommand(ComputeGraph.OP_HEAD_CACHE_WRITE, a.id, b.id, 2, 1, 1, 3, 3);
        graph.recordHeadPrefixMatmul(a, b, c, 2, 1, true);
        assertCommand(ComputeGraph.OP_HEAD_PREFIX_MATMUL, a.id, b.id, c.id, 2, 1, 3, 1, 3, 1);
        graph.recordHeadPrefixMatmul(a, b, c, 1, 2, false);
        assertCommand(ComputeGraph.OP_HEAD_PREFIX_MATMUL, a.id, b.id, c.id, 1, 2, 3, 2, 3, 0);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void rotaryRecordsTheInverseFlag(boolean inverse) {
        graph.recordRotary(a, b, c, d, 2, 3, 4, inverse);
        assertCommand(ComputeGraph.OP_ROTARY, a.id, b.id, c.id, d.id, 2, 3, 4, inverse ? 1 : 0);
    }

    @Test
    void scalarOperationsPreserveFloatBits() {
        graph.recordScalarUnary(ComputeGraph.OP_ADD_SCALAR, a, b, -0.25f);
        assertCommand(ComputeGraph.OP_ADD_SCALAR, a.id, b.id, bits(-0.25f), 6);
        graph.recordPow(a, b, 2.5f);
        assertCommand(ComputeGraph.OP_POW, a.id, b.id, bits(2.5f), 6);
        graph.recordClamp(a, b, -0.5f, 0.75f);
        assertCommand(ComputeGraph.OP_CLAMP, a.id, b.id, bits(-0.5f), bits(0.75f), 6);
    }

    @Test
    void scatterVariantsHaveDistinctOpcodes() {
        graph.recordScatterAddRows(a, b, c, 2, 3, 4);
        assertCommand(ComputeGraph.OP_SCATTER_ADD_ROWS, a.id, b.id, c.id, 2, 3, 4);
        graph.recordScatterAddRowsAtomic(a, b, c, 2, 3, 4);
        assertCommand(ComputeGraph.OP_SCATTER_ADD_ROWS_ATOMIC, a.id, b.id, c.id, 2, 3, 4);
    }

    @Test
    void lossCommandsPreserveTargetsAndSeparateOutputs() {
        graph.recordCrossEntropyLoss(a, b, c, 2, 3);
        assertCommand(ComputeGraph.OP_CROSS_ENTROPY_LOSS, a.id, b.id, c.id, 2, 3);
        graph.recordCrossEntropyGradient(a, b, c, 2, 3);
        assertCommand(ComputeGraph.OP_CROSS_ENTROPY_GRADIENT, a.id, b.id, c.id, 2, 3);
        graph.recordCrossEntropy(a, b, c, d, 2, 3);
        assertCommand(ComputeGraph.OP_CROSS_ENTROPY_FUSED, a.id, b.id, c.id, d.id, 2, 3);
    }

    @Test
    void rmsNormForwardAndBackwardPreserveCachedBuffers() {
        graph.recordRmsNorm(a, b, c, d, e, 2, 3, 1e-5f);
        assertCommand(ComputeGraph.OP_RMS_NORM, a.id, b.id, c.id, d.id, e.id, 2, 3, bits(1e-5f));
        graph.recordRmsNormBackward(a, b, c, d, e, 2, 3);
        assertCommand(ComputeGraph.OP_RMS_NORM_BACKWARD, a.id, b.id, c.id, d.id, e.id, 2, 3);
    }

    @Test
    void swiGluRecordsBothGradientOutputs() {
        graph.recordSwiGlu(a, b, c);
        assertCommand(ComputeGraph.OP_SWIGLU, a.id, b.id, c.id, 6);
        graph.recordSwiGluBackward(a, b, c, d, e);
        assertCommand(ComputeGraph.OP_SWIGLU_BACKWARD, a.id, b.id, c.id, d.id, e.id, 6);
    }

    @Test
    void broadcastAndReductionShapesAreNotInterchanged() {
        graph.recordRowBroadcast(ComputeGraph.OP_ADD_ROW_VECTOR, a, b, c, 2, 3);
        assertCommand(ComputeGraph.OP_ADD_ROW_VECTOR, a.id, b.id, c.id, 2, 3);
        graph.recordColBroadcast(ComputeGraph.OP_MULTIPLY_BROADCAST_COLS, a, b, c, 2, 3);
        assertCommand(ComputeGraph.OP_MULTIPLY_BROADCAST_COLS, a.id, b.id, c.id, 2, 3);
        graph.recordReduction(ComputeGraph.OP_MEAN_ALONG_ROWS, a, b, 2, 3);
        assertCommand(ComputeGraph.OP_MEAN_ALONG_ROWS, a.id, b.id, 2, 3);
    }

    @Test
    void scalarReductionsPreserveElementCounts() {
        graph.recordSumAbs(a, b, 2, 3);
        assertCommand(ComputeGraph.OP_SUM_ABS, a.id, b.id, 2, 3);
        graph.recordSumSquaresScalar(a, b);
        assertCommand(ComputeGraph.OP_SUM_SQUARES_SCALAR, a.id, b.id, 6);
        graph.recordSumScalar(a, b, 2, 3);
        assertCommand(ComputeGraph.OP_SUM_SCALAR, a.id, b.id, 2, 3);
        graph.recordTranspose(a, b, 2, 3);
        assertCommand(ComputeGraph.OP_TRANSPOSE, a.id, b.id, 2, 3);
    }

    @Test
    void adamWRecordsEveryHyperparameterInOrder() {
        graph.recordAdamWUpdate(a, b, c, d, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 6);
        assertCommand(ComputeGraph.OP_ADAMW_UPDATE, a.id, b.id, c.id, d.id,
                bits(0.1f), bits(0.2f), bits(0.3f), bits(0.4f), bits(0.5f), bits(0.6f), bits(0.7f), 6);
    }

    private void assertCommand(int... expected) {
        graph.flush();
        assertEquals(1, runtime.flushCalls.size());
        var call = runtime.flushCalls.get(0);
        assertEquals(expected.length, call.cmdStreamLength());
        assertArrayEquals(expected, Arrays.copyOf(call.cmdStream(), call.cmdStreamLength()));
        runtime.flushCalls.clear();
    }

    private static int bits(float value) {
        return Float.floatToRawIntBits(value);
    }
}
