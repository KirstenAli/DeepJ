package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Gemma BF16 eager attention rounds score intermediates before its FP32 softmax. */
final class Gemma2Attention {
    private Gemma2Attention() {}

    static Tensor apply(Tensor query, Tensor key, Tensor value, Tensor mask, float scale, float softcap) {
        if (query.dtype() != DType.BFLOAT16) return ModelOps.attention(query, key, value, mask, scale, softcap);
        return probabilities(query.matmul(key.transpose(-1, -2)), mask, scale, softcap).matmul(value);
    }

    static Tensor probabilities(Tensor scores, Tensor mask, float scale, float softcap) {
        DType dtype = scores.dtype();
        scores = scores.multiplyScalar(scale);
        if (softcap > 0) scores = scores.divideScalar(softcap).tanhActivation().multiplyScalar(softcap);
        if (mask != null) scores = scores.add(mask.to(dtype));
        return scores.to(DType.FLOAT32).softmax(-1).to(dtype);
    }
}
