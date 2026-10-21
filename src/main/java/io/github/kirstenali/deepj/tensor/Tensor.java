package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.autograd.*;

import java.util.Random;

public class Tensor {

    /** Unsafe storage access for backend kernels and initialization. Use set() for managed writes. */
    public final float[] data;
    public final int rows, cols;

    Object gpuTag;
    private boolean retainDeviceBuffer;
    private final GradientState gradientState = new GradientState();

    public GradientState gradientState() { return gradientState; }
    public boolean requiresGrad() { return gradientState.required(); }
    public Tensor requiresGrad(boolean required) {
        gradientState.require(required);
        return this;
    }
    public Tensor grad() { return gradientState.gradient(); }
    public void zeroGrad() { gradientState.bindGradient(null); }
    public void backward() { Autograd.backward(this); }
    public void backward(Tensor gradient) { Autograd.backward(this, gradient); }
    public Tensor detach() { return new Tensor(this); }
    public Tensor copy() { return UnaryOps.shift(this, 0.0f); }
    public Tensor sumTensor() { return ReductionOps.sum(this); }
    public float item() { return TensorStorage.item(this); }

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

    public Tensor matmul(Tensor other) { return BinaryOps.matmul(this, other); }
    public Tensor add(Tensor other) { return BinaryOps.add(this, other); }
    public Tensor subtract(Tensor other) { return BinaryOps.subtract(this, other); }
    public Tensor multiply(Tensor other) { return BinaryOps.multiply(this, other); }
    public Tensor divide(Tensor other) { return BinaryOps.divide(this, other); }

    public Tensor addRowVector(Tensor rowVector) { return BroadcastOps.add(this, rowVector, false); }
    public Tensor addBroadcastCols(Tensor colVector) { return BroadcastOps.add(this, colVector, true); }
    public Tensor divideBroadcastCols(Tensor colVector) { return BroadcastOps.divide(this, colVector); }
    public Tensor subtractBroadcastCols(Tensor colVector) { return BroadcastOps.subtract(this, colVector); }
    public Tensor multiplyBroadcastCols(Tensor colVector) { return BroadcastOps.multiply(this, colVector, true); }
    public Tensor addBroadcastRows(Tensor rowVector) { return BroadcastOps.add(this, rowVector, false); }
    public Tensor multiplyBroadcastRows(Tensor rowVector) { return BroadcastOps.multiply(this, rowVector, false); }

    public Tensor multiplyScalar(float s) { return UnaryOps.scale(this, s); }
    public Tensor addScalar(float s) { return UnaryOps.shift(this, s); }
    public Tensor divideScalar(float s) { return UnaryOps.divide(this, s); }

    public Tensor sumRows() { return ReductionOps.rows(this); }
    public Tensor sumAlongRows() { return ReductionOps.columns(this); }
    public Tensor sumAlongCols() { return ReductionOps.rows(this); }
    public Tensor meanAlongRows() { return ReductionOps.mean(this); }
    public Tensor varianceAlongRows() { return ReductionOps.variance(this); }

    public Tensor transpose() { return UnaryOps.transpose(this); }
    public Tensor sqrt() { return UnaryOps.sqrt(this); }
    public Tensor neg() { return UnaryOps.neg(this); }
    public Tensor exp() { return UnaryOps.exp(this); }
    public Tensor log() { return UnaryOps.log(this); }

    public Tensor tanhActivation() { return ActivationOps.tanh(this); }
    public Tensor sigmoidActivation() { return ActivationOps.sigmoid(this); }
    public Tensor reluActivation() { return ActivationOps.relu(this); }
    public Tensor geluActivation() { return ActivationOps.gelu(this); }

    public Tensor addInPlace(Tensor other) {
        return TensorMutation.apply(this, () -> backend().addInPlace(this, other), other);
    }

    public Tensor subtractInPlace(Tensor other) {
        return TensorMutation.apply(this, () -> backend().subtractInPlace(this, other), other);
    }

    public Tensor multiplyInPlace(Tensor other) {
        return TensorMutation.apply(this, () -> backend().multiplyInPlace(this, other), other);
    }

    public Tensor divideInPlace(Tensor other) {
        return TensorMutation.apply(this, () -> backend().divideInPlace(this, other), other);
    }

    public Tensor multiplyScalarInPlace(float scalar) {
        return TensorMutation.apply(this, () -> backend().multiplyScalarInPlace(this, scalar));
    }

    public Tensor addScalarInPlace(float scalar) {
        return TensorMutation.apply(this, () -> backend().addScalarInPlace(this, scalar));
    }

    public Tensor divideScalarInPlace(float scalar) {
        return TensorMutation.apply(this, () -> backend().divideScalarInPlace(this, scalar));
    }

    public Tensor sqrtInPlace() {
        return TensorMutation.apply(this, () -> backend().sqrtInPlace(this));
    }

    public Tensor negInPlace() {
        return TensorMutation.apply(this, () -> backend().negInPlace(this));
    }

    public Tensor expInPlace() {
        return TensorMutation.apply(this, () -> backend().expInPlace(this));
    }

    public Tensor logInPlace() {
        return TensorMutation.apply(this, () -> backend().logInPlace(this));
    }

    public Tensor reluInPlace() {
        return TensorMutation.apply(this, () -> backend().reluInPlace(this));
    }

    public Tensor geluInPlace() {
        return TensorMutation.apply(this, () -> backend().geluInPlace(this));
    }

    public Tensor tanhInPlace() {
        return TensorMutation.apply(this, () -> backend().tanhInPlace(this));
    }

    public Tensor sigmoidInPlace() {
        return TensorMutation.apply(this, () -> backend().sigmoidInPlace(this));
    }

    public Tensor softmaxRows() { return ActivationOps.softmax(this); }

    public static void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                                   float lr, float beta1, float beta2, float eps,
                                   float weightDecay, float bc1, float bc2) {
        w.gradientState.beforeMutation();
        mt.gradientState.beforeMutation();
        vt.gradientState.beforeMutation();
        backend().adamWUpdate(w, g, mt, vt, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
    }

    public Tensor maxAlongRows() {
        return SelectionOps.maximum(this);
    }

    public Tensor clamp(float min, float max) {
        return SelectionOps.clamp(this, min, max);
    }

    public Tensor pow(float exponent) {
        return UnaryOps.pow(this, exponent);
    }

    public static void scatterAddRows(Tensor target, int[] indices, Tensor grad) {
        target.gradientState.beforeMutation();
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
        gradientState.beforeMutation();
        TensorStorage.set(this, r, c, value);
    }

    public Tensor getRow(int row) {
        return sliceRows(this, new int[]{row}, cols);
    }

    public void setRow(int row, Tensor source, int srcRow) {
        gradientState.beforeMutation();
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
