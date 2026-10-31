package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

/** Historical matrix kernels remain the fast path inside the general final-axis contract. */
abstract class MetalNormalizationOps extends MetalGeneralNormalizationOps {
    @Override
    public Tensor layerNormBackward(Tensor dXHat, Tensor xHat, Tensor std, int dim) {
        Tensor.requireSameShape(dXHat, xHat, "layerNormBackward");
        if (dim != NormalizationChecks.width(xHat)) throw new IllegalArgumentException("LayerNorm width mismatch");
        NormalizationChecks.layerStatistics(xHat, std);
        if (!MetalDenseOps.sameMatrices(dXHat, xHat) || !MetalDenseOps.matrix(std)) {
            Tensor ones = addScalar(zeros(DType.FLOAT32, dim), 1);
            return normalizationBackward(dXHat, xHat, std, ones, true);
        }
        GpuBuffer gDXHat = gpuIn(dXHat), gXHat = gpuIn(xHat), gStd = gpuIn(std);
        GpuBuffer gOut = graph.newOutputBuffer(dXHat.matrixRows(), dXHat.matrixCols());
        graph.recordLayerNormBackward(gDXHat, gXHat, gStd, gOut, dXHat.matrixRows(), dXHat.matrixCols());
        return graph.createOutputLike(gOut, dXHat);
    }

    @Override
    public RmsNormResult rmsNorm(Tensor input, Tensor gamma, float epsilon) {
        NormalizationChecks.forward(input, gamma, null, epsilon, false);
        if (!MetalDenseOps.matrix(input) || !MetalDenseOps.matrix(gamma)) {
            var result = normalize(input, gamma, null, epsilon, false);
            return new RmsNormResult(result.output(), result.normalized(), result.denominator());
        }
        requireRmsNormShapes(input, gamma, input.matrixRows(), 1);
        GpuBuffer in = gpuIn(input), gain = gpuIn(gamma);
        GpuBuffer output = graph.newOutputBuffer(input.matrixRows(), input.matrixCols());
        GpuBuffer normalized = graph.newOutputBuffer(input.matrixRows(), input.matrixCols());
        GpuBuffer rms = graph.newOutputBuffer(input.matrixRows(), 1);
        graph.recordRmsNorm(in, gain, output, normalized, rms, input.matrixRows(), input.matrixCols(), epsilon);
        return new RmsNormResult(graph.createOutputLike(output, input), graph.createOutputLike(normalized, input),
                gpuOut(rms));
    }

    @Override
    public Tensor rmsNormBackward(Tensor gradient, Tensor normalized, Tensor rms, Tensor gamma) {
        Tensor.requireSameShape(gradient, normalized, "rmsNormBackward");
        if (!MetalDenseOps.sameMatrices(gradient, normalized) || !MetalDenseOps.matrix(rms)
                || !MetalDenseOps.matrix(gamma)) {
            return normalizationBackward(gradient, normalized, rms, gamma, false);
        }
        requireRmsNormShapes(gradient, gamma, rms.matrixRows(), rms.matrixCols());
        GpuBuffer g = gpuIn(gradient), norm = gpuIn(normalized), denominator = gpuIn(rms), gain = gpuIn(gamma);
        GpuBuffer output = graph.newOutputBuffer(gradient.matrixRows(), gradient.matrixCols());
        graph.recordRmsNormBackward(g, norm, denominator, gain, output, gradient.matrixRows(), gradient.matrixCols());
        return graph.createOutputLike(output, gradient);
    }

    static void requireRmsNormShapes(Tensor input, Tensor gamma, int rmsRows, int rmsCols) {
        if (gamma.matrixRows() != 1 || gamma.matrixCols() != input.matrixCols()) {
            throw new IllegalArgumentException("RMSNorm gamma must match input columns");
        }
        if (rmsRows != input.matrixRows() || rmsCols != 1) {
            throw new IllegalArgumentException("RMSNorm scale must match input rows");
        }
    }
}
