package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.nn.TensorJoin;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.LongBinaryOperator;
import java.util.function.UnaryOperator;

final class ArithmeticOperators {
    private ArithmeticOperators() {}

    static void register(OnnxRegistry r) {
        binary(r, "Add", Tensor::add, Math::addExact);
        binary(r, "Sub", Tensor::subtract, Math::subtractExact);
        binary(r, "Mul", Tensor::multiply, Math::multiplyExact);
        binary(r, "Div", Tensor::divide, Math::divideExact);
        unary(r, "Neg", Tensor::neg);
        unary(r, "Abs", ArithmeticOperators::abs);
        unary(r, "Exp", Tensor::exp);
        unary(r, "Log", Tensor::log);
        unary(r, "Sqrt", Tensor::sqrt);
        unary(r, "Reciprocal", x -> x.pow(-1));
        matrixAndVariadic(r);
        BooleanOperators.register(r);
    }

    private static void binary(OnnxRegistry r, String name, BinaryOperator<Tensor> floating, LongBinaryOperator host) {
        r.add(name, 7, "", c -> {
                    ValueOps.sameType(c.input(0), c.input(1));
                    OnnxValue result = c.input(0).isFloating() ? OnnxValue.of(floating.apply(c.tensor(0), c.tensor(1)))
                                    : ValueOps.binary(c.input(0), c.input(1), host);
                    return List.of(result);
                });
    }

    private static void unary(OnnxRegistry r, String name, UnaryOperator<Tensor> operation) {
        r.add(name, 6, "", c -> c.one(operation.apply(c.tensor(0))));
    }

    static Tensor abs(Tensor x) {
        return x.reluActivation().add(x.neg().reluActivation());
    }

    static Tensor maximum(Tensor a, Tensor b) {
        int[] shape = ValueOps.broadcast(a.shape(), b.shape());
        return TensorJoin.stack(0, a.expand(shape), b.expand(shape)).maximum(0, false);
    }

    private static Tensor matmul(OnnxContext c) {
        ValueOps.sameType(c.input(0), c.input(1));
        return c.tensor(0).matmul(c.tensor(1));
    }

    private static Tensor gemm(OnnxContext c) {
        Tensor a = c.integer("transA", 0) == 0 ? c.tensor(0) : c.tensor(0).transpose();
        Tensor b = c.integer("transB", 0) == 0 ? c.tensor(1) : c.tensor(1).transpose();
        if (a.rank() != 2 || b.rank() != 2)
            throw new IllegalArgumentException("Gemm inputs must be matrices");
        ValueOps.sameType(c.input(0), c.input(1));
        Tensor output = a.matmul(b).multiplyScalar(c.floating("alpha", 1));
        return c.hasInput(2) ? output.add(c.tensor(2).multiplyScalar(c.floating("beta", 1))) : output;
    }

    private static Tensor power(OnnxContext c) {
        Tensor exponent = c.tensor(1);
        if (exponent.numel() == 1 && !exponent.requiresGrad())
            return c.tensor(0).pow(exponent.item());
        throw new IllegalArgumentException("Pow currently requires a constant scalar exponent");
    }

    private static Tensor fold(OnnxContext c, BinaryOperator<Tensor> operation, boolean average) {
        Tensor output = c.tensor(0);
        for (int i = 1; i < c.inputCount(); i++) {
            output = operation.apply(output, c.tensor(i));
        }
        return average ? output.divideScalar(c.inputCount()) : output;
    }

    private static void matrixAndVariadic(OnnxRegistry r) {
        r.add("MatMul", 7, "", c -> c.one(matmul(c)));
        r.add("Gemm", 7, "alpha beta transA transB", c -> c.one(gemm(c)));
        r.add("Pow", 7, "", c -> c.one(power(c)));
        r.add("Sum", 8, "", c -> c.one(fold(c, Tensor::add, false)));
        r.add("Mean", 8, "", c -> c.one(fold(c, Tensor::add, true)));
        r.add("Max", 8, "", c -> c.one(fold(c, ArithmeticOperators::maximum, false)));
        r.add("Min", 8, "", c -> c.one(fold(c, (a, b) -> maximum(a.neg(), b.neg()).neg(), false)));
    }
}
