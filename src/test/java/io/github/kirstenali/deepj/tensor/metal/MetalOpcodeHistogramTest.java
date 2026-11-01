package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOrigin;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.GraphOpHistogram;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/** Manual: opcode histogram of one training step per dtype (needs -Ddeepj.graph.histogram=true). */
@Disabled("manual diagnostic")
class MetalOpcodeHistogramTest extends MetalDifferentialTestSupport {
    private static final DeepJOriginConfig CONFIG = new DeepJOriginConfig(1024, 128, 64, 4, 2, 256);

    @Test
    void opcodeCountsPerDtype() {
        Map<Integer, String> names = GraphOpHistogram.opcodeNames();
        Map<DType, Map<Integer, Long>> counts = new HashMap<>();
        for (DType dtype : new DType[]{DType.FLOAT32, DType.FLOAT16}) {
            counts.put(dtype, stepCounts(dtype));
        }
        for (int opcode : names.keySet()) {
            long f32 = counts.get(DType.FLOAT32).getOrDefault(opcode, 0L);
            long f16 = counts.get(DType.FLOAT16).getOrDefault(opcode, 0L);
            if (f32 != f16) System.out.printf("%-32s FLOAT32 %4d  FLOAT16 %4d%n", names.get(opcode), f32, f16);
        }
    }

    private Map<Integer, Long> stepCounts(DType dtype) {
        DeepJOrigin model = new DeepJOrigin(CONFIG, 7, dtype);
        AdamW optimizer = AdamW.defaultAdamW(1e-4f);
        int[] tokens = new Random(3).ints(128, 0, CONFIG.vocabSize()).toArray();
        step(model, optimizer, tokens);
        GraphOpHistogram.reset();
        step(model, optimizer, tokens);
        return GraphOpHistogram.snapshot();
    }

    private static void step(DeepJOrigin model, AdamW optimizer, int[] tokens) {
        model.parameters().forEach(p -> p.zeroGrad());
        Tensor loss = CrossEntropyLoss.forward(model.forward(tokens), tokens, null);
        loss.backward();
        optimizer.step(model.parameters());
        loss.item();
    }
}

