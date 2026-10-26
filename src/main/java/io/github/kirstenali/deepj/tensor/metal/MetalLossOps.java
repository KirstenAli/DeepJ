package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

/** Final-axis losses preserve matrix FP32 kernels and skip excluded rows before reading logits. */
abstract class MetalLossOps extends MetalNormalizationOps {
    @Override
    public Tensor crossEntropyGradient(Tensor logits, int[] targets) {
        CrossEntropyChecks.requireTargets(logits, targets);
        if (!denseLoss(logits, targets)) return graph.layoutCrossEntropy(logits, targets, null, 2).gradient();
        Tensor matrix = lossMatrix(logits), targetTensor = immutableIntColumn(targets);
        GpuBuffer input = gpuIn(matrix), ids = gpuIn(targetTensor);
        GpuBuffer output = graph.newOutputBuffer(matrix.matrixRows(), matrix.matrixCols());
        graph.recordCrossEntropyGradient(input, ids, output, matrix.matrixRows(), matrix.matrixCols());
        return reshape(gpuOut(output), logits.shape());
    }

    @Override
    public CrossEntropyResult crossEntropy(Tensor logits, int[] targets) {
        CrossEntropyChecks.requireTargets(logits, targets);
        if (!denseLoss(logits, targets)) return generalLoss(logits, targets, null);
        Tensor matrix = lossMatrix(logits), targetTensor = immutableIntColumn(targets);
        GpuBuffer input = gpuIn(matrix), ids = gpuIn(targetTensor);
        GpuBuffer losses = graph.newOutputBuffer(matrix.matrixRows(), 1);
        GpuBuffer gradient = graph.newOutputBuffer(matrix.matrixRows(), matrix.matrixCols());
        graph.recordCrossEntropy(input, ids, losses, gradient, matrix.matrixRows(), matrix.matrixCols());
        return new CrossEntropyResult(meanLoss(logits, gpuOut(losses), targets.length),
                reshape(gpuOut(gradient), logits.shape()));
    }

    @Override
    public float crossEntropyLoss(Tensor logits, int[] targets) {
        CrossEntropyChecks.requireTargets(logits, targets);
        Tensor rows = denseLoss(logits, targets) ? denseRowLosses(lossMatrix(logits), targets)
                : graph.layoutCrossEntropy(logits, targets, null, 1).loss();
        return meanLoss(logits, rows, targets.length).item();
    }

    @Override
    public float crossEntropyLoss(Tensor logits, int[] targets, boolean[] mask) {
        int count = maskedCount(logits, targets, mask);
        Tensor rows = graph.layoutCrossEntropy(logits, targets, mask, 1).loss();
        return meanLoss(logits, rows, count).item();
    }

    @Override
    public CrossEntropyResult crossEntropy(Tensor logits, int[] targets, boolean[] mask) {
        maskedCount(logits, targets, mask);
        return generalLoss(logits, targets, mask);
    }

    @Override
    public Tensor crossEntropyGradient(Tensor logits, int[] targets, boolean[] mask) {
        maskedCount(logits, targets, mask);
        return graph.layoutCrossEntropy(logits, targets, mask, 2).gradient();
    }

    private CrossEntropyResult generalLoss(Tensor logits, int[] targets, boolean[] mask) {
        int count = included(logits, targets, mask);
        var result = graph.layoutCrossEntropy(logits, targets, mask, 3);
        return new CrossEntropyResult(meanLoss(logits, result.loss(), count), result.gradient());
    }

    private Tensor denseRowLosses(Tensor matrix, int[] targets) {
        Tensor targetTensor = immutableIntColumn(targets);
        GpuBuffer input = gpuIn(matrix), ids = gpuIn(targetTensor);
        GpuBuffer output = graph.newOutputBuffer(matrix.matrixRows(), 1);
        graph.recordCrossEntropyLoss(input, ids, output, matrix.matrixRows(), matrix.matrixCols());
        return gpuOut(output);
    }

    private Tensor meanLoss(Tensor logits, Tensor losses, int included) {
        Tensor mean = divideScalar(sumRows(losses), included);
        return logits.rank() == 2 ? mean : reshape(mean, new int[0]);
    }

    private Tensor lossMatrix(Tensor logits) {
        return logits.rank() == 2 ? logits : reshape(logits, CrossEntropyChecks.positions(logits), logits.size(-1));
    }

    private static boolean denseLoss(Tensor logits, int[] targets) {
        return MetalDenseOps.dense(logits) && MetalIndexMetadata.fitsFloat(targets);
    }

    private static int included(Tensor logits, int[] targets, boolean[] mask) {
        CrossEntropyChecks.requireTargets(logits, targets);
        return mask == null ? targets.length : CrossEntropyChecks.included(mask, targets.length);
    }

    private static int maskedCount(Tensor logits, int[] targets, boolean[] mask) {
        CrossEntropyChecks.requireTargets(logits, targets);
        return CrossEntropyChecks.included(mask, targets.length);
    }
}
