package io.github.kirstenali.deepj.tensor;

import java.util.List;

public interface TensorBackend {

    /** Backend-owned positive-zero storage; ordinary Tensor constructors stay CPU-backed. */
    default Tensor zeros(int rows, int columns) { return new Tensor(rows, columns); }

    default Tensor zeros(DType dtype, int... shape) {
        if (dtype == DType.FLOAT32 && shape.length == 2) return zeros(shape[0], shape[1]);
        throw new UnsupportedOperationException("This backend does not support the requested zero tensor");
    }

    /** Independent storage with the same float bits, without recording an autograd operation. */
    default Tensor copy(Tensor source) { return new Tensor(source); }

    default Tensor view(Tensor input, int... shape) { return TensorViews.view(input, shape); }
    default Tensor unsqueeze(Tensor input, int axis) { return TensorViews.unsqueeze(input, axis); }
    default Tensor permute(Tensor input, int... axes) { return TensorViews.permute(input, axes); }
    default Tensor select(Tensor input, int axis, int index) { return TensorViews.select(input, axis, index); }
    default Tensor slice(Tensor input, int axis, int start, int stop, int step) {
        return TensorViews.slice(input, axis, start, stop, step);
    }
    default Tensor expand(Tensor input, int... shape) { return TensorViews.expand(input, shape); }

    default Tensor reshape(Tensor input, int... shape) {
        return view(input.isContiguous() ? input : copy(input), shape);
    }

    default Tensor cast(Tensor input, DType dtype) {
        if (input.dtype() == dtype) return input;
        throw new UnsupportedOperationException("This backend does not support the requested dtype conversion");
    }

    default Tensor sumAxes(Tensor input, int[] axes, boolean keepDimensions) {
        throw new UnsupportedOperationException("This backend does not support axis reductions");
    }

    default Tensor meanAxes(Tensor input, int[] axes, boolean keepDimensions) {
        throw new UnsupportedOperationException("This backend does not support axis means");
    }

    default Tensor varianceAxes(Tensor input, int[] axes, boolean keepDimensions, int correction) {
        throw new UnsupportedOperationException("This backend does not support axis variance");
    }

    default Tensor maximumAxes(Tensor input, int[] axes, boolean keepDimensions) {
        throw new UnsupportedOperationException("This backend does not support axis maxima");
    }

    default Tensor maximumAxesBackward(Tensor input, Tensor maximum, Tensor gradient, int[] axes, boolean keep) {
        throw new UnsupportedOperationException("This backend does not support axis maximum gradients");
    }

    default Tensor sumToShape(Tensor input, int... shape) { return GradientShapes.sumTo(this, input, shape); }

    default Tensor sliceGradient(Tensor gradient, int[] shape, int axis, int start, int step) {
        throw new UnsupportedOperationException("This backend does not support slice gradients");
    }

    default Tensor indexSelect(Tensor input, int axis, int[] indices) {
        throw new UnsupportedOperationException("This backend does not support ND index selection");
    }

    default void indexAdd(Tensor target, int axis, int[] indices, Tensor updates) {
        throw new UnsupportedOperationException("This backend does not support ND indexed updates");
    }

    default Tensor indexSelectGradient(Tensor gradient, int[] shape, int axis, int[] indices) {
        throw new UnsupportedOperationException("This backend does not support ND index selection gradients");
    }

    default Tensor softmaxAxis(Tensor input, int axis) {
        throw new UnsupportedOperationException("This backend does not support axis softmax");
    }

    default Tensor softmaxAxisBackward(Tensor gradient, Tensor output, int axis) {
        Tensor product = multiply(gradient, output);
        return multiply(output, subtract(gradient, sumAxes(product, new int[]{axis}, true)));
    }

    Tensor matmul(Tensor a, Tensor b);

    default void writeHeadCache(Tensor cache, Tensor update, int heads, int offset) {
        HeadCacheOps.write(cache, update, heads, offset);
    }

    default Tensor headPrefixMatmul(Tensor left, Tensor cache, int heads, int length, boolean transposeCache) {
        return batchedMatmul(left, HeadCacheOps.prefix(cache, heads, length), heads, false, transposeCache);
    }

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
        if (batches == 1) {
            left.materialize();
            right.materialize();
            Tensor a = transposeLeft ? transpose(left) : left;
            Tensor b = transposeRight ? transpose(right) : right;
            return matmul(a, b);
        }
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
            for (float value : tensor.rawData()) {
                squares += value * value;
            }
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

    default Tensor softmaxBackward(Tensor gradOutput, Tensor softmaxOut, float scale) {
        return multiplyScalar(softmaxBackward(gradOutput, softmaxOut), scale);
    }

    float crossEntropyLoss(Tensor logits, int[] targets);
    Tensor crossEntropyGradient(Tensor logits, int[] targets);

    default CrossEntropyResult crossEntropy(Tensor logits, int[] targets) {
        return CrossEntropyDefaults.apply(this, logits, targets, null);
    }

    float crossEntropyLoss(Tensor logits, int[] targets, boolean[] mask);
    Tensor crossEntropyGradient(Tensor logits, int[] targets, boolean[] mask);

    default CrossEntropyResult crossEntropy(Tensor logits, int[] targets, boolean[] mask) {
        return CrossEntropyDefaults.apply(this, logits, targets, mask);
    }

    void adamWUpdate(Tensor w, Tensor g, Tensor mt, Tensor vt,
                     float lr, float beta1, float beta2, float eps,
                     float weightDecay, float bc1, float bc2);

    Tensor layerNormBackward(Tensor dXHat, Tensor xHat, Tensor std, int dim);

    default NormalizationResult normalize(Tensor input, Tensor gamma, Tensor beta, float epsilon, boolean centered) {
        throw new UnsupportedOperationException("This backend does not support ND normalization");
    }

    default Tensor normalizationBackward(Tensor gradient, Tensor normalized, Tensor denominator,
                                            Tensor gamma, boolean centered) {
        throw new UnsupportedOperationException("This backend does not support ND normalization gradients");
    }

    default RmsNormResult rmsNorm(Tensor input, Tensor gamma, float epsilon) {
        Tensor meanSquare = meanAlongRows(multiply(input, input));
        Tensor rms = sqrt(addScalar(meanSquare, epsilon));
        Tensor normalized = divideBroadcastCols(input, rms);
        return new RmsNormResult(multiplyBroadcastRows(normalized, gamma), normalized, rms);
    }

    default Tensor rmsNormBackward(Tensor gradient, Tensor normalized,
                                   Tensor rms, Tensor gamma) {
        Tensor scaled = multiplyBroadcastRows(gradient, gamma);
        Tensor inner = meanAlongRows(multiply(scaled, normalized));
        return divideBroadcastCols(subtract(scaled,
                multiplyBroadcastCols(normalized, inner)), rms);
    }

    default Tensor swiGlu(Tensor gate, Tensor up) {
        Tensor activated = multiply(gate, sigmoid(gate));
        return multiply(activated, up);
    }

    default SwiGluBackwardResult swiGluBackward(Tensor gradient, Tensor gate,
                                                Tensor up) {
        Tensor sigmoid = sigmoid(gate);
        Tensor activated = multiply(gate, sigmoid);
        Tensor complement = addScalar(multiplyScalar(sigmoid, -1.0f), 1.0f);
        Tensor derivative = add(sigmoid, multiply(activated, complement));
        Tensor gateGradient = multiply(multiply(gradient, up), derivative);
        return new SwiGluBackwardResult(gateGradient, multiply(gradient, activated));
    }

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

    default void materializeTensor(Tensor t) {}

    default void releaseTemporaryResources() { releaseResources(); }

    default void releaseResources() {}
}
