package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.layers.BiasFreeLinear;
import io.github.kirstenali.deepj.layers.Projection;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
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
        this(dModel, dFF, rnd, bias, ParameterFactory.float32());
    }

    public SwiGLULayer(int dModel, int dFF, Random rnd, boolean bias, ParameterFactory parameters) {
        if (dModel <= 0) throw new IllegalArgumentException("dModel must be > 0");
        if (dFF    <= 0) throw new IllegalArgumentException("dFF must be > 0");

        this.gateProj = projection(dModel, dFF, rnd, bias, parameters.scope("gate"));
        this.upProj   = projection(dModel, dFF, rnd, bias, parameters.scope("up"));
        this.downProj = projection(dFF, dModel, rnd, bias, parameters.scope("down"));
    }

    private static Projection projection(int input, int output, Random random, boolean bias,
                                         ParameterFactory parameters) {
        return bias ? new Linear(input, output, random, parameters)
                : new BiasFreeLinear(input, output, random, parameters);
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
