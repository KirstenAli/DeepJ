package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.HeadLayout;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Rotary pairs connect the first and second halves of each head. */
public final class HalfSplitRotary {
    private HalfSplitRotary() {}

    public static Tensor apply(Tensor input, Tensor cosine, Tensor sine, int length) {
        if (cosine.requiresGrad() || sine.requiresGrad()) {
            throw new IllegalArgumentException("Rotary tables must be constants");
        }
        if (input.rank() == 4) {
            if (input.size(2) != length) throw new IllegalArgumentException("Rotary time axis mismatch");
            return apply(HeadLayout.pack(input), cosine, sine, length).reshape(input.shape());
        }
        Tensor output = Tensor.backend().rotaryHalfSplit(input, cosine, sine, length, false);
        return Autograd.recordBuiltinSaved(output,
                g -> new Tensor[]{Tensor.backend().rotaryHalfSplit(g, cosine, sine, length, true)},
                new Tensor[]{input}, cosine, sine);
    }
}
