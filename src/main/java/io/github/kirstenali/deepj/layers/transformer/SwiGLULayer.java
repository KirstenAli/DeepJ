package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.layers.BiasFreeLinear;
import io.github.kirstenali.deepj.layers.Projection;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.autograd.ActivationOps;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class SwiGLULayer implements Layer {

    private final Projection gateProj;
    private final Projection upProj;
    private final Projection downProj;

    public SwiGLULayer(int dModel, int dFF, Random rnd) {
        this(dModel, dFF, rnd, true);
    }

    public SwiGLULayer(int dModel, int dFF, Random rnd, boolean bias) {
        if (dModel <= 0) throw new IllegalArgumentException("dModel must be > 0");
        if (dFF    <= 0) throw new IllegalArgumentException("dFF must be > 0");

        this.gateProj = projection(dModel, dFF, rnd, bias);
        this.upProj   = projection(dModel, dFF, rnd, bias);
        this.downProj = projection(dFF, dModel, rnd, bias);
    }

    private static Projection projection(int input, int output, Random random, boolean bias) {
        return bias ? new Linear(input, output, random) : new BiasFreeLinear(input, output, random);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor gate = gateProj.forward(x);
        Tensor up = upProj.forward(x);
        return downProj.forward(ActivationOps.swiGlu(gate, up));
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
