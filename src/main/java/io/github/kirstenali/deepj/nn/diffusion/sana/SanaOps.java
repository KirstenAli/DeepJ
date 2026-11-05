package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.nn.TensorJoin;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.stream.IntStream;

/** Model arithmetic composes existing differentiable DeepJ tensor operations. */
final class SanaOps {
    private SanaOps() {}

    static Tensor linear(Tensor input, SanaWeights weights, String name) {
        Tensor result = input.matmul(weights.tensor(name + ".weight").transpose(-1, -2));
        return weights.contains(name + ".bias") ? result.add(weights.tensor(name + ".bias")) : result;
    }

    static Tensor silu(Tensor input) { return input.multiply(input.sigmoidActivation()); }

    static Tensor rms(Tensor input, Tensor gain, Tensor bias, float epsilon, boolean gemma) {
        Tensor value = input.to(DType.FLOAT32);
        Tensor result = value.divide(value.multiply(value).mean(-1, true).addScalar(epsilon).sqrt());
        if (gain != null) result = result.multiply(gemma ? gain.to(DType.FLOAT32).addScalar(1) : gain);
        if (bias != null) result = result.add(bias);
        return result.to(input.dtype());
    }

    static Tensor norm(Tensor input, float epsilon) {
        Tensor value = input.to(DType.FLOAT32);
        Tensor centered = value.subtract(value.mean(-1, true));
        return centered.divide(centered.multiply(centered).mean(-1, true).addScalar(epsilon).sqrt())
                .to(input.dtype());
    }

    static Tensor affineRms(Tensor input, SanaWeights weights, String name, float epsilon, boolean gemma) {
        Tensor bias = weights.contains(name + ".bias") ? weights.tensor(name + ".bias") : null;
        return rms(input, weights.tensor(name + ".weight"), bias, epsilon, gemma);
    }

    static Tensor heads(Tensor input, int heads) {
        return input.reshape(input.size(0), input.size(1), heads, input.size(2) / heads).permute(0, 2, 1, 3);
    }

    static Tensor mergeHeads(Tensor input) {
        return input.permute(0, 2, 1, 3).reshape(input.size(0), input.size(2), input.size(1) * input.size(3));
    }

    static Tensor imageTokens(Tensor input) {
        return input.permute(0, 2, 3, 1).reshape(input.size(0), input.size(2) * input.size(3), input.size(1));
    }

    static Tensor image(Tensor input, int height, int width) {
        return input.reshape(input.size(0), height, width, input.size(2)).permute(0, 3, 1, 2);
    }

    static Tensor repeat(Tensor input, int axis, int repeats) {
        int[] indices = IntStream.range(0, Math.multiplyExact(input.size(axis), repeats))
                .map(i -> i / repeats).toArray();
        return input.indexSelect(axis, indices);
    }

    static Tensor upsample(Tensor input) { return repeat(repeat(input, 2, 2), 3, 2); }

    static Tensor pixelShuffle(Tensor input) {
        return input.reshape(input.size(0), input.size(1) / 4, 2, 2, input.size(2), input.size(3))
                .permute(0, 1, 4, 2, 5, 3).reshape(input.size(0), input.size(1) / 4,
                        input.size(2) * 2, input.size(3) * 2);
    }

    static Tensor linearAttention(Tensor query, Tensor key, Tensor value, float epsilon) {
        Tensor q = query.to(DType.FLOAT32).reluActivation(), k = key.to(DType.FLOAT32).reluActivation();
        Tensor result = q.matmul(k.transpose(-1, -2).matmul(value.to(DType.FLOAT32)));
        Tensor divisor = q.matmul(k.sum(-2, true).transpose(-1, -2)).addScalar(epsilon);
        return result.divide(divisor).to(value.dtype());
    }

    static Tensor attention(Tensor query, Tensor key, Tensor value, Tensor mask, float scale, float softcap) {
        Tensor scores = query.matmul(key.transpose(-1, -2)).to(DType.FLOAT32).multiplyScalar(scale);
        if (softcap > 0) scores = scores.divideScalar(softcap).tanhActivation().multiplyScalar(softcap);
        if (mask != null) scores = scores.add(mask);
        return scores.softmax(-1).to(value.dtype()).matmul(value);
    }

    static Tensor cat(int axis, Tensor... inputs) { return TensorJoin.concatenate(axis, inputs); }
}
