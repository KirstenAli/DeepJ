package io.github.kirstenali.deepj.tensor;

import java.util.List;

public interface TensorBackend {

    Tensor matmul(Tensor a, Tensor b);

    default Tensor sliceRows(Tensor input, int[] rows) {
        return TensorBackendDefaults.sliceRows(input, rows);
    }

    default Tensor splitHeads(Tensor input, int heads) {
        return TensorBackendDefaults.splitHeads(input, heads);
    }

    default Tensor mergeHeads(Tensor input, int heads) {
        return TensorBackendDefaults.mergeHeads(input, heads);
    }

    default Tensor batchedMatmul(Tensor left, Tensor right, int batches,
                                 boolean transposeLeft, boolean transposeRight) {
        return TensorBackendDefaults.batchedMatmul(
                left, right, batches, transposeLeft, transposeRight);
    }

    default Tensor causalMask(Tensor input, int sequenceLength) {
        return TensorBackendDefaults.causalMask(input, sequenceLength);
    }

    default Tensor causalSoftmax(Tensor input, int sequenceLength, float scale) {
        Tensor scaled = multiplyScalar(input, scale);
        return softmaxRows(causalMask(scaled, sequenceLength));
    }

    default Tensor rotary(Tensor input, Tensor cosine, Tensor sine,
                          int sequenceLength, boolean inverse) {
        return TensorBackendDefaults.rotary(input, cosine, sine, sequenceLength, inverse);
    }

    Tensor add(Tensor a, Tensor b);
    Tensor subtract(Tensor a, Tensor b);
    Tensor multiply(Tensor a, Tensor b);
    Tensor divide(Tensor a, Tensor b);

    Tensor addRowVector(Tensor a, Tensor rowVector);

    Tensor addBroadcastCols(Tensor a, Tensor colVector);
    Tensor divideBroadcastCols(Tensor a, Tensor colVector);
    Tensor subtractBroadcastCols(Tensor a, Tensor colVector);
    Tensor multiplyBroadcastCols(Tensor a, Tensor colVector);

    Tensor addBroadcastRows(Tensor a, Tensor rowVector);
    Tensor multiplyBroadcastRows(Tensor a, Tensor rowVector);

    Tensor multiplyScalar(Tensor a, float scalar);
    Tensor addScalar(Tensor a, float scalar);
    Tensor divideScalar(Tensor a, float scalar);

    Tensor sumRows(Tensor a);
    Tensor sumAlongRows(Tensor a);
    Tensor sumAlongCols(Tensor a);
    Tensor meanAlongRows(Tensor a);
    Tensor varianceAlongRows(Tensor a);
    Tensor maxAlongRows(Tensor a);
    float sum(Tensor a);
    float sumAbs(Tensor a);

    default float l2Norm(List<Tensor> tensors) {
        float squares = 0.0f;
        for (Tensor tensor : tensors) {
            tensor.materialize();
            for (float value : tensor.data) squares += value * value;
        }
        return (float) Math.sqrt(squares);
    }

    Tensor transpose(Tensor a);
    Tensor clamp(Tensor a, float min, float max);
    Tensor sqrt(Tensor a);
    Tensor pow(Tensor a, float exponent);
    Tensor neg(Tensor a);
    Tensor exp(Tensor a);
    Tensor log(Tensor a);

    Tensor tanh(Tensor a);
    Tensor sigmoid(Tensor a);
    Tensor relu(Tensor a);
    Tensor reluBackward(Tensor input, Tensor gradOutput);
    Tensor gelu(Tensor a);
    Tensor geluBackward(Tensor input, Tensor gradOutput);

    Tensor softmaxRows(Tensor logits);
    Tensor softmaxBackward(Tensor gradOutput, Tensor softmaxOut);

    float crossEntropyLoss(Tensor logits, int[] targets);
    Tensor crossEntropyGradient(Tensor logits, int[] targets);
    float crossEntropyLoss(Tensor logits, int[] targets, boolean[] mask);
    Tensor crossEntropyGradient(Tensor logits, int[] targets, boolean[] mask);

    void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                     float lr, float beta1, float beta2, float eps,
                     float weightDecay, float bc1, float bc2);

    Tensor layerNormBackward(Tensor dXHat, Tensor xHat, Tensor std, int dim);

    void scatterAddRows(Tensor target, int[] indices, Tensor grad);

    void addInPlace(Tensor a, Tensor b);
    void subtractInPlace(Tensor a, Tensor b);
    void multiplyInPlace(Tensor a, Tensor b);
    void divideInPlace(Tensor a, Tensor b);

    void multiplyScalarInPlace(Tensor a, float s);
    void addScalarInPlace(Tensor a, float s);
    void divideScalarInPlace(Tensor a, float s);

    void sqrtInPlace(Tensor a);
    void negInPlace(Tensor a);
    void expInPlace(Tensor a);
    void logInPlace(Tensor a);
    void reluInPlace(Tensor a);
    void geluInPlace(Tensor a);
    void tanhInPlace(Tensor a);
    void sigmoidInPlace(Tensor a);

    default void materializeTensor(Tensor t) {  }

    default void releaseTemporaryResources() { releaseResources(); }

    default void releaseResources() {  }
}
