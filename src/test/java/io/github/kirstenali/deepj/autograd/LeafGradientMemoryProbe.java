package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/** Leaf VJP temporaries must not survive the whole backward pass. */
public final class LeafGradientMemoryProbe {
    private LeafGradientMemoryProbe() {}

    public static void main(String[] args) {
        List<Parameter> parameters = parameters();
        Tensor input = Tensor.ones(1, 1).requiresGrad(true);
        Tensor output = input;
        for (Parameter parameter : parameters) {
            output = append(output, parameter);
        }
        output.backward();
        verify(parameters, input);
        System.out.println("Leaf gradient memory regression passed");
    }

    private static List<Parameter> parameters() {
        List<Parameter> result = new ArrayList<>();
        for (int index = 0; index < 48; index++) {
            result.add(new Parameter(Tensor.ones(256, 1024)));
        }
        return result;
    }

    private static Tensor append(Tensor previous, Parameter parameter) {
        return Autograd.recordSaved(Tensor.ones(1, 1),
                gradient -> new Tensor[]{gradient, Tensor.ones(256, 1024)},
                new Tensor[]{previous, parameter.value});
    }

    private static void verify(List<Parameter> parameters, Tensor input) {
        if (input.grad().item() != 1) throw new AssertionError("Incorrect root gradient");
        for (Parameter parameter : parameters) {
            for (float value : parameter.grad().rawData()) {
                if (value != 1) throw new AssertionError("Incorrect leaf gradient");
            }
        }
    }
}
