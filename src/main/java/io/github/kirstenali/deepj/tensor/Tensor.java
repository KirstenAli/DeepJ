package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;

import java.util.Random;

public class Tensor {

    public final float[] data;
    public final int rows, cols;

    Object gpuTag;
    private boolean retainDeviceBuffer;

    public Object getGpuTag() { return gpuTag; }

    public void setGpuTag(Object tag) { this.gpuTag = tag; }

    public Tensor retainDeviceBuffer() {
        retainDeviceBuffer = true;
        return this;
    }

    public boolean retainsDeviceBuffer() { return retainDeviceBuffer; }

    private static volatile TensorBackend BACKEND = new CpuBackend();
    public static void setBackend(TensorBackend backend) {
        if (backend == null) throw new IllegalArgumentException("backend cannot be null");
        BACKEND = backend;
    }

    public static TensorBackend backend() {
        return BACKEND;
    }

    public Tensor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new float[TensorStorage.checkedSize(rows, cols)];
    }

    public Tensor(Tensor source) {
        TensorStorage.requireSource(source);
        this.rows = source.rows;
        this.cols = source.cols;
        this.data = TensorStorage.copyData(source);
        this.gpuTag = null;
    }

    public float[] rowData(int r) {
        return TensorStorage.rowData(this, r);
    }

    public void materialize() {
        TensorStorage.materialize(this);
    }

    public Tensor matmul(Tensor other) { return backend().matmul(this, other); }
    public Tensor add(Tensor other) { return backend().add(this, other); }
    public Tensor subtract(Tensor other) { return backend().subtract(this, other); }
    public Tensor multiply(Tensor other) { return backend().multiply(this, other); }
    public Tensor divide(Tensor other) { return backend().divide(this, other); }

    public Tensor addRowVector(Tensor rowVector) { return backend().addRowVector(this, rowVector); }
    public Tensor addBroadcastCols(Tensor colVector) { return backend().addBroadcastCols(this, colVector); }
    public Tensor divideBroadcastCols(Tensor colVector) { return backend().divideBroadcastCols(this, colVector); }
    public Tensor subtractBroadcastCols(Tensor colVector) { return backend().subtractBroadcastCols(this, colVector); }
    public Tensor multiplyBroadcastCols(Tensor colVector) { return backend().multiplyBroadcastCols(this, colVector); }
    public Tensor addBroadcastRows(Tensor rowVector) { return backend().addBroadcastRows(this, rowVector); }
    public Tensor multiplyBroadcastRows(Tensor rowVector) { return backend().multiplyBroadcastRows(this, rowVector); }

    public Tensor multiplyScalar(float s) { return backend().multiplyScalar(this, s); }
    public Tensor addScalar(float s) { return backend().addScalar(this, s); }
    public Tensor divideScalar(float s) { return backend().divideScalar(this, s); }

    public Tensor sumRows() { return backend().sumRows(this); }
    public Tensor sumAlongRows() { return backend().sumAlongRows(this); }
    public Tensor sumAlongCols() { return backend().sumAlongCols(this); }
    public Tensor meanAlongRows() { return backend().meanAlongRows(this); }
    public Tensor varianceAlongRows() { return backend().varianceAlongRows(this); }

    public Tensor transpose() { return backend().transpose(this); }
    public Tensor sqrt() { return backend().sqrt(this); }
    public Tensor neg() { return backend().neg(this); }
    public Tensor exp() { return backend().exp(this); }
    public Tensor log() { return backend().log(this); }

    public Tensor tanhActivation() { return backend().tanh(this); }
    public Tensor sigmoidActivation() { return backend().sigmoid(this); }
    public Tensor reluActivation() { return backend().relu(this); }
    public Tensor reluBackward(Tensor gradOutput) { return backend().reluBackward(this, gradOutput); }
    public Tensor geluActivation() { return backend().gelu(this); }
    public Tensor geluBackward(Tensor gradOutput) { return backend().geluBackward(this, gradOutput); }

    public Tensor addInPlace(Tensor other) {
        backend().addInPlace(this, other);
        return this;
    }

    public Tensor subtractInPlace(Tensor other) {
        backend().subtractInPlace(this, other);
        return this;
    }

    public Tensor multiplyInPlace(Tensor other) {
        backend().multiplyInPlace(this, other);
        return this;
    }

    public Tensor divideInPlace(Tensor other) {
        backend().divideInPlace(this, other);
        return this;
    }

    public Tensor multiplyScalarInPlace(float scalar) {
        backend().multiplyScalarInPlace(this, scalar);
        return this;
    }

    public Tensor addScalarInPlace(float scalar) {
        backend().addScalarInPlace(this, scalar);
        return this;
    }

    public Tensor divideScalarInPlace(float scalar) {
        backend().divideScalarInPlace(this, scalar);
        return this;
    }

    public Tensor sqrtInPlace() {
        backend().sqrtInPlace(this);
        return this;
    }

    public Tensor negInPlace() {
        backend().negInPlace(this);
        return this;
    }

    public Tensor expInPlace() {
        backend().expInPlace(this);
        return this;
    }

    public Tensor logInPlace() {
        backend().logInPlace(this);
        return this;
    }

    public Tensor reluInPlace() {
        backend().reluInPlace(this);
        return this;
    }

    public Tensor geluInPlace() {
        backend().geluInPlace(this);
        return this;
    }

    public Tensor tanhInPlace() {
        backend().tanhInPlace(this);
        return this;
    }

    public Tensor sigmoidInPlace() {
        backend().sigmoidInPlace(this);
        return this;
    }

    public Tensor softmaxRows() { return backend().softmaxRows(this); }
    public Tensor softmaxBackward(Tensor softmaxOut) { return backend().softmaxBackward(this, softmaxOut); }

    public Tensor crossEntropyGradient(int[] targets) { return backend().crossEntropyGradient(this, targets); }

    public static void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                                   float lr, float beta1, float beta2, float eps,
                                   float weightDecay, float bc1, float bc2) {
        backend().adamWUpdate(w, g, mt, vt, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
    }

    public static Tensor layerNormBackward(Tensor dXHat, Tensor xHat, Tensor std, int dim) {
        return backend().layerNormBackward(dXHat, xHat, std, dim);
    }

    public Tensor maxAlongRows() {
        return backend().maxAlongRows(this);
    }

    public Tensor clamp(float min, float max) {
        return backend().clamp(this, min, max);
    }

    public Tensor pow(float exponent) {
        return backend().pow(this, exponent);
    }

    public static void scatterAddRows(Tensor target, int[] indices, Tensor grad) {
        backend().scatterAddRows(target, indices, grad);
    }

    public static Tensor from2D(float[][] data) {
        return TensorStorage.from2D(data);
    }

    public float sum() {
        return backend().sum(this);
    }

    public float sumAbs() {
        return backend().sumAbs(this);
    }

    public float crossEntropyLoss(int[] targets) {
        return backend().crossEntropyLoss(this, targets);
    }

    public float get(int r, int c) {
        return TensorStorage.get(this, r, c);
    }

    public void set(int r, int c, float value) {
        TensorStorage.set(this, r, c, value);
    }

    public Tensor getRow(int row) {
        return TensorStorage.getRow(this, row);
    }

    public void setRow(int row, Tensor source, int srcRow) {
        TensorStorage.setRow(this, row, source, srcRow);
    }

    public static Tensor sliceRows(Tensor t, int[] rowIndices, int cols) {
        return TensorStorage.sliceRows(t, rowIndices, cols);
    }

    public static Tensor sampleRows(Tensor t, int n, Random rnd) {
        return TensorStorage.sampleRows(t, n, rnd);
    }

    public void print(String label) {
        TensorStorage.print(this, label);
    }

    public static Tensor zeros(int rows, int cols) { return TensorStorage.zeros(rows, cols); }
    public static Tensor ones(int rows, int cols) { return TensorStorage.ones(rows, cols); }
    public static Tensor random(int rows, int cols, Random rand) { return TensorStorage.random(rows, cols, rand); }
    public static Tensor causalMask(int size) { return TensorStorage.causalMask(size); }

    public static void requireSameShape(Tensor a, Tensor b, String op) {
        TensorStorage.requireSameShape(a, b, op);
    }

    public static void requireTargetsMatchRows(Tensor logits, int[] targets) {
        TensorStorage.requireTargetsMatchRows(logits, targets);
    }
}
