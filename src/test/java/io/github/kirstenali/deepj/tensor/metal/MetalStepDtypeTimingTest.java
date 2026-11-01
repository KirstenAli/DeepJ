package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOrigin;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.Locale;
import java.util.Random;

/** Exploratory Origin timing, not qualification. ALL rotates dtype order; BACKEND=cpu selects CPU. */
@Disabled("manual timing")
class MetalStepDtypeTimingTest extends MetalDifferentialTestSupport {
    private static final int WARMUP = setting("DEEPJ_TIMING_WARMUP", 10);
    private static final int STEPS = setting("DEEPJ_TIMING_STEPS", 30), ROUNDS = 3;
    private static final int SEQUENCE = setting("DEEPJ_TIMING_SEQUENCE", 1024);
    private static final int WIDTH = setting("DEEPJ_TIMING_WIDTH", 768);
    private static final DeepJOriginConfig CONFIG = new DeepJOriginConfig(setting("DEEPJ_TIMING_VOCAB", 16384),
            SEQUENCE, WIDTH, 12, setting("DEEPJ_TIMING_LAYERS", 12), 4 * WIDTH);

    @Test
    void stepTimingForConfiguredDtype() {
        if (System.getenv().getOrDefault("DEEPJ_TIMING_BACKEND", "metal").equals("cpu")) Tensor.setBackend(cpu);
        String selected = System.getenv().getOrDefault("DEEPJ_TIMING_DTYPE", "FLOAT32");
        DType[] dtypes = selected.equals("ALL") ? DType.values() : new DType[]{DType.valueOf(selected)};
        int rounds = dtypes.length == 1 ? 1 : ROUNDS;
        for (int round = 0; round < rounds; round++) {
            for (int index = 0; index < dtypes.length; index++) {
                DType dtype = dtypes[(index + round) % dtypes.length];
                System.out.printf(Locale.ROOT, "round %d %-9s %s layers=%d width=%d seq=%d vocab=%d %8.1f ms/step%n",
                        round, dtype, Tensor.backend().getClass().getSimpleName(), CONFIG.nLayers(), WIDTH, SEQUENCE,
                        CONFIG.vocabSize(), time(dtype));
            }
        }
    }

    private double time(DType dtype) {
        DeepJOrigin model = new DeepJOrigin(CONFIG, 7, dtype);
        AdamW optimizer = AdamW.defaultAdamW(1e-4f);
        int[] tokens = tokens();
        for (int i = 0; i < WARMUP; i++) {
            step(model, optimizer, tokens);
        }
        long start = System.nanoTime();
        for (int i = 0; i < STEPS; i++) {
            step(model, optimizer, tokens);
        }
        long elapsed = System.nanoTime() - start;
        System.gc();
        return elapsed / 1e6 / STEPS;
    }

    private static void step(DeepJOrigin model, AdamW optimizer, int[] tokens) {
        model.parameters().forEach(p -> p.zeroGrad());
        Tensor loss = CrossEntropyLoss.forward(model.forward(tokens), tokens, null);
        loss.backward();
        optimizer.step(model.parameters());
        loss.item();
    }

    private static int[] tokens() {
        return new Random(3).ints(SEQUENCE, 0, CONFIG.vocabSize()).toArray();
    }

    private static int setting(String name, int fallback) {
        return Integer.parseInt(System.getenv().getOrDefault(name, Integer.toString(fallback)));
    }
}

