package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.data.BatchSource;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.models.CausalLM;
import io.github.kirstenali.deepj.tensor.Tensor;

public final class CausalLMEvaluation {

    private CausalLMEvaluation() {}

    public static EvaluationResult evaluate(CausalLM model, BatchSource source,
                                            int batches, int batchSize) {
        validate(model, source, batches, batchSize);
        double lossSum = 0.0;
        long sequenceCount = (long) batches * batchSize;
        long tokens = 0L;
        try {
            for (int i = 0; i < batches; i++) {
                Batch batch = source.nextBatch(batchSize);
                lossSum += batchLoss(model, batch);
                tokens += batchTokenCount(batch);
            }
        } finally {
            Tensor.backend().releaseResources();
        }
        double loss = lossSum / sequenceCount;
        return new EvaluationResult(loss, Math.exp(loss), tokens);
    }

    private static double batchLoss(CausalLM model, Batch batch) {
        double loss = 0.0;
        for (int row = 0; row < batch.x().length; row++) {
            Tensor logits = model.forward(batch.x()[row]);
            loss += sequenceLoss(logits, batch.y()[row], batch.mask(row));
        }
        return loss;
    }

    private static long batchTokenCount(Batch batch) {
        long count = 0;
        for (int row = 0; row < batch.x().length; row++) {
            count += tokenCount(batch, row);
        }
        return count;
    }

    private static float sequenceLoss(Tensor logits, int[] targets, boolean[] mask) {
        return mask == null ? CrossEntropyLoss.loss(logits, targets)
                : CrossEntropyLoss.loss(logits, targets, mask);
    }

    private static int tokenCount(Batch batch, int row) {
        boolean[] mask = batch.mask(row);
        if (mask == null) return batch.x()[row].length;
        int count = 0;
        for (boolean included : mask) {
            if (included) count++;
        }
        return count;
    }

    private static void validate(CausalLM model, BatchSource source, int batches, int batchSize) {
        if (model == null || source == null) {
            throw new IllegalArgumentException("model and source must not be null");
        }
        if (batches <= 0 || batchSize <= 0) {
            throw new IllegalArgumentException("batches and batchSize must be positive");
        }
    }
}
