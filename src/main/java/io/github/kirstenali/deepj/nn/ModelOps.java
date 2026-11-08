package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.autograd.ActivationOps;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.stream.IntStream;

/** Model arithmetic composes existing differentiable DeepJ tensor operations. */
public final class ModelOps {
    private ModelOps() {}

    public static Tensor linear(Tensor input, TensorWeights weights, String name) {
        Tensor result = project(input, weights.tensor(name + ".weight").transpose(-1, -2));
        return weights.contains(name + ".bias") ? result.add(weights.tensor(name + ".bias")) : result;
    }

    /** Projects independent rows with a right-hand [input, output] matrix, preserving leading dimensions. */
    public static Tensor project(Tensor input, Tensor weight) {
        if (!compactBatch(input, weight)) return input.matmul(weight);
        int[] shape = input.shape();
        shape[shape.length - 1] = weight.size(-1);
        return input.reshape(-1, input.size(-1)).matmul(weight).reshape(shape);
    }

    private static boolean compactBatch(Tensor input, Tensor weight) {
        return !GradMode.isEnabled() && Tensor.backend().usesDeviceStorage()
                && input.dtype() == DType.BFLOAT16 && weight.dtype() == DType.BFLOAT16 && weight.rank() == 2
                && weight.size(-1) >= 256 && weight.size(-1) % 8 == 0
                && independentRows(input);
    }

    private static boolean independentRows(Tensor input) {
        return input.rank() > 2 && input.isContiguous() && input.storageOffset() == 0
                && input.size(-2) >= 64 && input.size(-1) >= 256
                && input.numel() > (long) input.size(-2) * input.size(-1);
    }

    public static Tensor silu(Tensor input) {
        if (input.dtype() == DType.FLOAT32) return input.multiply(input.sigmoidActivation());
        Tensor unit = Tensor.backend().usesDeviceStorage()
                ? Tensor.backend().expand(Tensor.from(new float[]{1}, input.dtype(), new int[0]), input.shape())
                : Tensor.ones(input.dtype(), input.shape());
        return ActivationOps.swiGlu(input, unit);
    }

    public static Tensor rms(Tensor input, Tensor gain, Tensor bias, float epsilon, boolean gemma) {
        if (ModelRms.supports(input, gain, epsilon)) return ModelRms.inference(input, gain, bias, epsilon, gemma);
        Tensor value = input.to(DType.FLOAT32);
        Tensor result = value.divide(value.multiply(value).mean(-1, true).addScalar(epsilon).sqrt());
        if (gain != null && !gemma && gain.dtype() != DType.FLOAT32) result = result.to(gain.dtype());
        if (gain != null) result = result.multiply(gemma ? gain.to(DType.FLOAT32).addScalar(1) : gain);
        if (bias != null) result = result.add(bias);
        return result.to(input.dtype());
    }

    public static Tensor norm(Tensor input, float epsilon) {
        if (ModelNorm.supports(input, epsilon)) return ModelNorm.inference(input, epsilon);
        Tensor value = input.to(DType.FLOAT32);
        Tensor centered = value.subtract(value.mean(-1, true));
        return centered.divide(centered.multiply(centered).mean(-1, true).addScalar(epsilon).sqrt())
                .to(input.dtype());
    }

    public static Tensor affineRms(Tensor input, TensorWeights weights, String name, float epsilon, boolean gemma) {
        Tensor bias = weights.contains(name + ".bias") ? weights.tensor(name + ".bias") : null;
        return rms(input, weights.tensor(name + ".weight"), bias, epsilon, gemma);
    }

    public static Tensor heads(Tensor input, int heads) {
        return input.reshape(input.size(0), input.size(1), heads, input.size(2) / heads).permute(0, 2, 1, 3);
    }

    public static Tensor mergeHeads(Tensor input) {
        return input.permute(0, 2, 1, 3).reshape(input.size(0), input.size(2), input.size(1) * input.size(3));
    }

    public static Tensor imageTokens(Tensor input) {
        return input.permute(0, 2, 3, 1).reshape(input.size(0), input.size(2) * input.size(3), input.size(1));
    }

    public static Tensor image(Tensor input, int height, int width) {
        return input.reshape(input.size(0), height, width, input.size(2)).permute(0, 3, 1, 2);
    }

    public static Tensor repeat(Tensor input, int axis, int repeats) {
        int[] indices = IntStream.range(0, Math.multiplyExact(input.size(axis), repeats))
                .map(i -> i / repeats).toArray();
        return input.indexSelect(axis, indices);
    }

    public static Tensor upsample(Tensor input) { return repeat(repeat(input, 2, 2), 3, 2); }

    public static Tensor pixelShuffle(Tensor input) {
        return input.reshape(input.size(0), input.size(1) / 4, 2, 2, input.size(2), input.size(3))
                .permute(0, 1, 4, 2, 5, 3).reshape(input.size(0), input.size(1) / 4,
                        input.size(2) * 2, input.size(3) * 2);
    }

    public static Tensor linearAttention(Tensor query, Tensor key, Tensor value, float epsilon) {
        Tensor q = query.to(DType.FLOAT32).reluActivation(), k = key.to(DType.FLOAT32).reluActivation();
        if (!GradMode.isEnabled()) {
            return ModelLinearAttention.apply(q, k, value, epsilon);
        }
        Tensor result = q.matmul(k.transpose(-1, -2).matmul(value.to(DType.FLOAT32)));
        Tensor divisor = q.matmul(k.sum(-2, true).transpose(-1, -2)).addScalar(epsilon);
        return result.divide(divisor).to(value.dtype());
    }

    public static Tensor attention(Tensor query, Tensor key, Tensor value, Tensor mask, float scale, float softcap) {
        Tensor scores = query.matmul(key.transpose(-1, -2)).to(DType.FLOAT32).multiplyScalar(scale);
        if (softcap > 0) scores = scores.divideScalar(softcap).tanhActivation().multiplyScalar(softcap);
        if (mask != null) scores = scores.add(mask);
        return scores.softmax(-1).to(value.dtype()).matmul(value);
    }

    public static Tensor cat(int axis, Tensor... inputs) { return TensorJoin.concatenate(axis, inputs); }
}
