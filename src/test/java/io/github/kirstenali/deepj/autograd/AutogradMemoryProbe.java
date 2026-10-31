package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

/** A small fixed-heap process makes activation-retention regressions deterministic. */
public final class AutogradMemoryProbe {
    private AutogradMemoryProbe() {}

    public static void main(String[] args) {
        Tensor input = Tensor.ones(512, 1_024).requiresGrad(true);
        Tensor output = switch (args[0]) {
            case "shift" -> shifts(input);
            case "softmax" -> softmaxBranches(input);
            default -> throw new IllegalArgumentException("Unknown memory probe");
        };
        output.backward(Tensor.ones(output.matrixRows(), output.matrixCols()));
        float expectedGradient = args[0].equals("shift") ? 1 : 0;
        if (input.grad().get(0, 0) != expectedGradient) {
            throw new AssertionError("Incorrect gradient after releasing unused values");
        }
        System.out.println("Memory regression passed");
    }

    private static Tensor shifts(Tensor input) {
        Tensor output = input;
        for (int index = 0; index < 128; index++) {
            output = output.addScalar(1);
        }
        if (output.get(0, 0) != 129) throw new AssertionError("Incorrect output");
        return output;
    }

    private static Tensor softmaxBranches(Tensor input) {
        Tensor output = Tensor.zeros(1, 1);
        for (int index = 0; index < 20; index++) {
            output = output.add(input.addScalar(index).softmaxRows().sumTensor());
        }
        if (output.item() != 20 * input.matrixRows()) throw new AssertionError("Incorrect output");
        return output;
    }
}
