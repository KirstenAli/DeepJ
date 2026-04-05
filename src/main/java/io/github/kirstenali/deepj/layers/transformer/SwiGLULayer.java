package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.activations.SiLU;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * SwiGLU feed-forward layer — the FFN used in Llama, Mistral, Qwen, and DeepSeek.
 *
 * <p>Unlike the standard two-projection MLP, SwiGLU uses <em>three</em> projections and a
 * multiplicative gate:
 * <pre>
 *   gate   = gateProj(x)              [seqLen × dFF]
 *   up     = upProj(x)                [seqLen × dFF]
 *   fused  = SiLU(gate) · up          [seqLen × dFF]   (element-wise)
 *   out    = downProj(fused)           [seqLen × dModel]
 * </pre>
 *
 * <p>Backward (chain rule through the gated multiply):
 * <pre>
 *   dFused     = downProj.backward(dOut)
 *   dSiluGate  = dFused · up           → fed into SiLU.backward → dGate
 *   dUp        = dFused · SiLU(gate)   → fed into upProj.backward
 *   dX         = gateProj.backward(dGate) + upProj.backward(dUp)
 * </pre>
 *
 * <p>Used as the feed-forward component in {@link io.github.kirstenali.deepj.layers.transformer.blocks.LlamaTransformerBlock}
 * and {@link io.github.kirstenali.deepj.layers.transformer.blocks.DeepSeekTransformerBlock}.
 */
public final class SwiGLULayer implements Layer {

    private final Linear gateProj;   // dModel → dFF
    private final Linear upProj;     // dModel → dFF
    private final Linear downProj;   // dFF   → dModel
    private final SiLU   silu;

    // Forward cache
    private Tensor siluGate;   // SiLU(gateProj(x))  [seqLen × dFF]
    private Tensor upOut;      // upProj(x)            [seqLen × dFF]

    /**
     * @param dModel input and output dimension
     * @param dFF    intermediate (hidden) dimension — typically ≈ 8/3 × dModel for Llama style
     * @param rnd    random source for weight initialisation
     */
    public SwiGLULayer(int dModel, int dFF, Random rnd) {
        if (dModel <= 0) throw new IllegalArgumentException("dModel must be > 0");
        if (dFF    <= 0) throw new IllegalArgumentException("dFF must be > 0");

        this.gateProj = new Linear(dModel, dFF, rnd);
        this.upProj   = new Linear(dModel, dFF, rnd);
        this.downProj = new Linear(dFF, dModel, rnd);
        this.silu     = new SiLU();
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor gate = gateProj.forward(x);   // [seqLen × dFF]
        this.upOut    = upProj.forward(x);   // [seqLen × dFF]
        this.siluGate = silu.forward(gate);  // SiLU(gate)

        Tensor fused = siluGate.multiply(upOut);  // element-wise gating
        return downProj.forward(fused);
    }

    @Override
    public Tensor backward(Tensor dOut) {
        // Gradient through downProj: dFused = dOut · W_down^T  [seqLen × dFF]
        Tensor dFused = downProj.backward(dOut);

        // Gate branch: upstream grad to SiLU = dFused · upOut
        Tensor dSiluGate = dFused.multiply(upOut);
        Tensor dGate = silu.backward(dSiluGate);

        // Up branch: upstream grad to upProj = dFused · siluGate
        Tensor dUp = dFused.multiply(siluGate);

        // Accumulate projection grads and return combined dX
        Tensor dXFromGate = gateProj.backward(dGate);
        Tensor dXFromUp   = upProj.backward(dUp);
        return dXFromGate.add(dXFromUp);
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        ps.addAll(gateProj.parameters());
        ps.addAll(upProj.parameters());
        ps.addAll(downProj.parameters());
        return ps;
    }
}

