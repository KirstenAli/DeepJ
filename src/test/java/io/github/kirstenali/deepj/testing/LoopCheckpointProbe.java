package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.data.StatefulBatchSource;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoop;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoopConfig;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import io.github.kirstenali.deepj.persistence.TrainingCheckpoint;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.Trainer;
import io.github.kirstenali.deepj.training.TrainingProgress;

import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.stream.IntStream;

/** Optional full-size Metal resume/memory probe. Removes only its own generated checkpoint. */
public final class LoopCheckpointProbe {

    private LoopCheckpointProbe() {}

    public static void main(String[] args) throws Exception {
        if (!MetalBackend.isAvailable()) throw new IllegalStateException("This probe requires Metal");
        Tensor.setBackend(new MetalBackend());
        Path checkpoint = Files.createTempFile("deepj-loop-resume-", ".dj");
        System.out.println("Temporary diagnostic checkpoint: " + checkpoint);
        try {
            probe(checkpoint);
        } finally {
            Files.deleteIfExists(checkpoint);
            Tensor.backend().releaseResources();
        }
    }

    private static void probe(Path checkpoint) throws Exception {
        var config = new DeepJLoopConfig(16_384, 1_024, 576, 9, 3, 6, 12, 6, 2, 1_664);
        var model = new DeepJLoop(config, 100L);
        var data = new ProbeData();
        var optimizer = AdamW.defaultAdamW(0.0001f);
        var schedule = new CosineLearningRateSchedule(0.0001f, 0.00001f, 2_000, 2_000_000);
        Trainer trainer = CausalLMTraining.trainer(model, data, optimizer);
        float initial = train(trainer);
        TrainingCheckpoint.save(model, optimizer, data,
                new TrainingProgress(1, initial, initial), schedule, checkpoint);
        Snapshot expected = snapshot(trainer, model, optimizer);
        TrainingCheckpoint.load(model, optimizer, data, schedule, checkpoint);
        Snapshot actual = snapshot(trainer, model, optimizer);
        if (!expected.equals(actual)) {
            throw new AssertionError("Resume mismatch: " + expected + " versus " + actual);
        }
        System.out.println("RESULT fullSizeCheckpointResume=EXACT " + actual);
    }

    private static float train(Trainer trainer) {
        float loss = trainer.trainStep(1);
        Tensor.backend().releaseTemporaryResources();
        return loss;
    }

    private static Snapshot snapshot(Trainer trainer, DeepJLoop model, AdamW optimizer) throws Exception {
        float loss = train(trainer);
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        for (var parameter : model.parameters()) {
            update(digest, parameter.value);
        }
        return new Snapshot(loss, HexFormat.of().formatHex(digest.digest()), optimizer.stepCount());
    }

    private static void update(MessageDigest digest, Tensor tensor) {
        tensor.materialize();
        ByteBuffer buffer = ByteBuffer.allocate(65_536);
        for (float value : tensor.rawData()) {
            if (!buffer.hasRemaining()) {
                digest.update(buffer.array());
                buffer.clear();
            }
            buffer.putFloat(value);
        }
        digest.update(buffer.array(), 0, buffer.position());
    }

    private record Snapshot(float loss, String weightSha256, long optimizerStep) {}

    private static final class ProbeData implements StatefulBatchSource {
        private long cursor;

        @Override
        public Batch nextBatch(int size) {
            if (size != 1) throw new IllegalArgumentException("This probe uses batch size one");
            int offset = Math.toIntExact(cursor++);
            return new Batch(new int[][]{sequence(offset)}, new int[][]{sequence(offset + 1)});
        }

        private static int[] sequence(int offset) {
            return IntStream.range(0, 1_024).map(index -> ((index + offset) * 37 + 11) % 16_384).toArray();
        }

        @Override
        public long randomState() {
            return cursor;
        }

        @Override
        public void restoreRandomState(long state) {
            if (state < 0) throw new IllegalArgumentException("Negative probe cursor");
            cursor = state;
        }
    }
}
