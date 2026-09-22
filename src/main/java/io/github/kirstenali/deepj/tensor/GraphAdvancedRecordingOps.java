package io.github.kirstenali.deepj.tensor;

import java.lang.ref.WeakReference;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

abstract class GraphAdvancedRecordingOps extends GraphRecordingOps {
    GraphAdvancedRecordingOps(GpuRuntime runtime) {
        super(runtime);
    }
    public void recordCrossEntropy(GpuBuffer logits, GpuBuffer targets,
                                   GpuBuffer losses, GpuBuffer gradient,
                                   int rows, int cols) {
        beginOp(7);
        emitInt(OP_CROSS_ENTROPY_FUSED);
        emitInt(logits.id);
        emitInt(targets.id);
        emitInt(losses.id);
        emitInt(gradient.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordRmsNorm(GpuBuffer input, GpuBuffer gamma, GpuBuffer output,
                              GpuBuffer normalized, GpuBuffer rms,
                              int rows, int cols, float epsilon) {
        beginOp(9);
        emitInt(OP_RMS_NORM);
        emitInt(input.id);
        emitInt(gamma.id);
        emitInt(output.id);
        emitInt(normalized.id);
        emitInt(rms.id);
        emitInt(rows);
        emitInt(cols);
        emitFloatBits(epsilon);
        endOp();
    }

    public void recordRmsNormBackward(GpuBuffer gradient, GpuBuffer normalized,
                                      GpuBuffer rms, GpuBuffer gamma,
                                      GpuBuffer output, int rows, int cols) {
        beginOp(8);
        emitInt(OP_RMS_NORM_BACKWARD);
        emitInt(gradient.id);
        emitInt(normalized.id);
        emitInt(rms.id);
        emitInt(gamma.id);
        emitInt(output.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordSwiGlu(GpuBuffer gate, GpuBuffer up, GpuBuffer fused) {
        beginOp(5);
        emitInt(OP_SWIGLU);
        emitInt(gate.id);
        emitInt(up.id);
        emitInt(fused.id);
        emitInt(gate.floatCount());
        endOp();
    }

    public void recordSwiGluBackward(GpuBuffer gradient, GpuBuffer gate, GpuBuffer up,
                                     GpuBuffer gateGradient, GpuBuffer upGradient) {
        beginOp(7);
        emitInt(OP_SWIGLU_BACKWARD);
        emitInt(gradient.id);
        emitInt(gate.id);
        emitInt(up.id);
        emitInt(gateGradient.id);
        emitInt(upGradient.id);
        emitInt(gradient.floatCount());
        endOp();
    }

    public void recordSumScalar(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(OP_SUM_SCALAR);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordTranspose(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(OP_TRANSPOSE);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordRowBroadcast(int opCode, GpuBuffer a, GpuBuffer rowVec, GpuBuffer out, int rows, int cols) {
        beginOp(6);
        emitInt(opCode);
        emitInt(a.id);
        emitInt(rowVec.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordColBroadcast(int opCode, GpuBuffer a, GpuBuffer colVec, GpuBuffer out, int rows, int cols) {
        beginOp(6);
        emitInt(opCode);
        emitInt(a.id);
        emitInt(colVec.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordReduction(int opCode, GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(opCode);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordSoftmaxRows(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(OP_SOFTMAX_ROWS);
        emitInt(in.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordSoftmaxBackward(GpuBuffer gradOutput, GpuBuffer softmaxOut, GpuBuffer out, int rows, int cols) {
        beginOp(6);
        emitInt(OP_SOFTMAX_BACKWARD);
        emitInt(gradOutput.id);
        emitInt(softmaxOut.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordLayerNormBackward(GpuBuffer dXHat, GpuBuffer xHat, GpuBuffer std,
                                        GpuBuffer out, int rows, int cols) {
        beginOp(7);
        emitInt(OP_LAYERNORM_BACKWARD);
        emitInt(dXHat.id);
        emitInt(xHat.id);
        emitInt(std.id);
        emitInt(out.id);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordAdamWUpdate(GpuBuffer w, GpuBuffer g, GpuBuffer mt, GpuBuffer vt,
                                  float lr, float beta1, float beta2, float eps,
                                  float weightDecay, float bc1, float bc2, int n) {
        beginOp(13);
        emitInt(OP_ADAMW_UPDATE);
        emitInt(w.id);
        emitInt(g.id);
        emitInt(mt.id);
        emitInt(vt.id);
        emitFloatBits(lr);
        emitFloatBits(beta1);
        emitFloatBits(beta2);
        emitFloatBits(eps);
        emitFloatBits(weightDecay);
        emitFloatBits(bc1);
        emitFloatBits(bc2);
        emitInt(n);
        endOp();
    }


}
