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
        emitBuffer(logits);
        emitBuffer(targets);
        emitBuffer(losses);
        emitBuffer(gradient);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordRmsNorm(GpuBuffer input, GpuBuffer gamma, GpuBuffer output,
                              GpuBuffer normalized, GpuBuffer rms,
                              int rows, int cols, float epsilon) {
        beginOp(9);
        emitInt(OP_RMS_NORM);
        emitBuffer(input);
        emitBuffer(gamma);
        emitBuffer(output);
        emitBuffer(normalized);
        emitBuffer(rms);
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
        emitBuffer(gradient);
        emitBuffer(normalized);
        emitBuffer(rms);
        emitBuffer(gamma);
        emitBuffer(output);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordSwiGlu(GpuBuffer gate, GpuBuffer up, GpuBuffer fused) {
        beginOp(5);
        emitInt(OP_SWIGLU);
        emitBuffer(gate);
        emitBuffer(up);
        emitBuffer(fused);
        emitInt(gate.floatCount());
        endOp();
    }

    public void recordSwiGluBackward(GpuBuffer gradient, GpuBuffer gate, GpuBuffer up,
                                     GpuBuffer gateGradient, GpuBuffer upGradient) {
        beginOp(7);
        emitInt(OP_SWIGLU_BACKWARD);
        emitBuffer(gradient);
        emitBuffer(gate);
        emitBuffer(up);
        emitBuffer(gateGradient);
        emitBuffer(upGradient);
        emitInt(gradient.floatCount());
        endOp();
    }

    public void recordSumScalar(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(OP_SUM_SCALAR);
        emitBuffer(in);
        emitBuffer(out);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordTranspose(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(OP_TRANSPOSE);
        emitBuffer(in);
        emitBuffer(out);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordRowBroadcast(int opCode, GpuBuffer a, GpuBuffer rowVec, GpuBuffer out, int rows, int cols) {
        beginOp(6);
        emitInt(opCode);
        emitBuffer(a);
        emitBuffer(rowVec);
        emitBuffer(out);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordColBroadcast(int opCode, GpuBuffer a, GpuBuffer colVec, GpuBuffer out, int rows, int cols) {
        beginOp(6);
        emitInt(opCode);
        emitBuffer(a);
        emitBuffer(colVec);
        emitBuffer(out);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordReduction(int opCode, GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(opCode);
        emitBuffer(in);
        emitBuffer(out);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordSoftmaxRows(GpuBuffer in, GpuBuffer out, int rows, int cols) {
        beginOp(5);
        emitInt(OP_SOFTMAX_ROWS);
        emitBuffer(in);
        emitBuffer(out);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordSoftmaxBackward(GpuBuffer gradOutput, GpuBuffer softmaxOut, GpuBuffer out, int rows, int cols) {
        beginOp(6);
        emitInt(OP_SOFTMAX_BACKWARD);
        emitBuffer(gradOutput);
        emitBuffer(softmaxOut);
        emitBuffer(out);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordScaledSoftmaxBackward(GpuBuffer gradient, GpuBuffer probabilities,
                                            GpuBuffer output, float scale) {
        beginOp(7);
        emitInt(OP_SCALED_SOFTMAX_BACKWARD);
        emitBuffer(gradient);
        emitBuffer(probabilities);
        emitBuffer(output);
        emitInt(gradient.rows);
        emitInt(gradient.cols);
        emitFloatBits(scale);
        endOp();
    }

    public void recordLayerNormBackward(GpuBuffer dXHat, GpuBuffer xHat, GpuBuffer std,
                                        GpuBuffer out, int rows, int cols) {
        beginOp(7);
        emitInt(OP_LAYERNORM_BACKWARD);
        emitBuffer(dXHat);
        emitBuffer(xHat);
        emitBuffer(std);
        emitBuffer(out);
        emitInt(rows);
        emitInt(cols);
        endOp();
    }

    public void recordAdamWUpdate(GpuBuffer w, GpuBuffer g, GpuBuffer mt, GpuBuffer vt,
                                  float lr, float beta1, float beta2, float eps,
                                  float weightDecay, float bc1, float bc2, int n) {
        beginOp(13);
        emitInt(OP_ADAMW_UPDATE);
        emitBuffer(w);
        emitBuffer(g);
        emitBuffer(mt);
        emitBuffer(vt);
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
