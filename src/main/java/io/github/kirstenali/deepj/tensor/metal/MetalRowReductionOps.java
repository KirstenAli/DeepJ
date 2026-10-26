package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;

import java.util.List;

abstract class MetalRowReductionOps extends MetalAxisOps {
    @Override
    public Tensor sumRows(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return MetalNdReductions.matrix(graph, a, 0, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(1, a.matrixCols());
        graph.recordReduction(ComputeGraph.OP_SUM_ROWS, ga, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor sumAlongRows(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return MetalNdReductions.matrix(graph, a, 1, 0);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), 1);
        graph.recordReduction(ComputeGraph.OP_SUM_ALONG_ROWS, ga, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor sumAlongCols(Tensor a) {
        return sumRows(a);
    }

    @Override
    public Tensor meanAlongRows(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return MetalNdReductions.matrix(graph, a, 1, 1);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), 1);
        graph.recordReduction(ComputeGraph.OP_MEAN_ALONG_ROWS, ga, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor varianceAlongRows(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return MetalNdReductions.matrix(graph, a, 1, 2);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), 1);
        graph.recordReduction(ComputeGraph.OP_VARIANCE_ALONG_ROWS, ga, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public Tensor maxAlongRows(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return MetalNdReductions.matrix(graph, a, 1, 3);
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixRows(), 1);
        graph.recordReduction(ComputeGraph.OP_MAX_ALONG_ROWS, ga, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

    @Override
    public float sum(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutTotal(a, 0).item();
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(1, 1);
        graph.recordSumScalar(ga, gOut, a.matrixRows(), a.matrixCols());
        Tensor scalar = gpuOut(gOut);
        scalar.materialize();
        return scalar.rawData()[0];
    }

    @Override
    public float sumAbs(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.layoutTotal(a, 4).item();
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gRowAbsSums = graph.newOutputBuffer(a.matrixRows(), 1);
        graph.recordSumAbs(ga, gRowAbsSums, a.matrixRows(), a.matrixCols());
        Tensor rowAbsSums = gpuOut(gRowAbsSums);
        Tensor scalar = sumRows(rowAbsSums);
        scalar.materialize();
        return scalar.rawData()[0];
    }

    @Override
    public float l2Norm(List<Tensor> tensors) {
        if (tensors.isEmpty()) return 0.0f;
        if (!tensors.stream().allMatch(MetalDenseOps::matrix)) return MetalNdReductions.norm(graph, tensors);
        Tensor total = new Tensor(1, 1);
        GpuBuffer output = gpuIn(total);
        for (Tensor tensor : tensors) {
            graph.recordSumSquaresScalar(gpuIn(tensor), output);
        }
        output.cpuStale = true;
        total.materialize();
        return (float) Math.sqrt(total.rawData()[0]);
    }

    @Override
    public Tensor transpose(Tensor a) {
        if (!MetalDenseOps.matrix(a)) return graph.copyToDevice(permute(a, 1, 0));
        GpuBuffer ga = gpuIn(a);
        GpuBuffer gOut = graph.newOutputBuffer(a.matrixCols(), a.matrixRows());
        graph.recordTranspose(ga, gOut, a.matrixRows(), a.matrixCols());
        return gpuOut(gOut);
    }

}
