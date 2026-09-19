package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.SwiGluBackwardResult;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class SwiGLULayer implements Layer {

    private final Linear gateProj;
    private final Linear upProj;
    private final Linear downProj;

    private Tensor gateOut;
    private Tensor upOut;

    public SwiGLULayer(int dModel, int dFF, Random rnd) {
        if (dModel <= 0) throw new IllegalArgumentException("dModel must be > 0");
        if (dFF    <= 0) throw new IllegalArgumentException("dFF must be > 0");

        this.gateProj = new Linear(dModel, dFF, rnd);
        this.upProj   = new Linear(dModel, dFF, rnd);
        this.downProj = new Linear(dFF, dModel, rnd);
    }

    @Override
    public Tensor forward(Tensor x) {
        gateOut = gateProj.forward(x);
        upOut = upProj.forward(x);
        return downProj.forward(Tensor.backend().swiGlu(gateOut, upOut));
    }

    @Override
    public Tensor backward(Tensor dOut) {

        Tensor dFused = downProj.backward(dOut);

        SwiGluBackwardResult gradients = Tensor.backend()
                .swiGluBackward(dFused, gateOut, upOut);
        Tensor dXFromGate = gateProj.backward(gradients.gateGradient());
        Tensor dXFromUp = upProj.backward(gradients.upGradient());
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
