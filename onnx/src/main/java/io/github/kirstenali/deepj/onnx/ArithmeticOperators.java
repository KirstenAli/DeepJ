package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.nn.TensorJoin;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.LongBinaryOperator;
import java.util.function.UnaryOperator;

final class ArithmeticOperators {
    private ArithmeticOperators() {}

    static void register(OnnxRegistry registry) {
        binary(registry, "Add", Tensor::add, Math::addExact);
        binary(registry, "Sub", Tensor::subtract, Math::subtractExact);
        binary(registry, "Mul", Tensor::multiply, Math::multiplyExact);
        binary(registry, "Div", Tensor::divide, Math::divideExact);
        unary(registry, "Neg", Tensor::neg);
        unary(registry, "Abs", ArithmeticOperators::abs);
        unary(registry, "Exp", Tensor::exp);
        unary(registry, "Log", Tensor::log);
        unary(registry, "Sqrt", Tensor::sqrt);
        unary(registry, "Reciprocal", x -> x.pow(-1));
        matrixAndVariadic(registry);
        BooleanOperators.register(registry);
    }

    private static void binary(
            OnnxRegistry registry,
            String name,
            BinaryOperator<Tensor> floating,
            LongBinaryOperator host) {
        registry.add(
                name,
                7,
                "",
                context -> {
                    ValueOps.sameType(context.input(0), context.input(1));
                    OnnxValue result =
                            context.input(0).isFloating()
                                    ? OnnxValue.of(
                                            floating.apply(context.tensor(0), context.tensor(1)))
                                    : ValueOps.binary(context.input(0), context.input(1), host);
                    return List.of(result);
                });
    }

    private static void unary(OnnxRegistry registry, String name, UnaryOperator<Tensor> operation) {
        registry.add(name, 6, "", context -> context.one(operation.apply(context.tensor(0))));
    }

    static Tensor abs(Tensor x) {
        return x.reluActivation().add(x.neg().reluActivation());
    }

    static Tensor maximum(Tensor a, Tensor b) {
        int[] shape = ValueOps.broadcast(a.shape(), b.shape());
        return TensorJoin.stack(0, a.expand(shape), b.expand(shape)).maximum(0, false);
    }

    private static Tensor matmul(OnnxContext context) {
        ValueOps.sameType(context.input(0), context.input(1));
        return context.tensor(0).matmul(context.tensor(1));
    }

    private static Tensor gemm(OnnxContext context) {
        Tensor a =
                context.integer("transA", 0) == 0
                        ? context.tensor(0)
                        : context.tensor(0).transpose();
        Tensor b =
                context.integer("transB", 0) == 0
                        ? context.tensor(1)
                        : context.tensor(1).transpose();
        if (a.rank() != 2 || b.rank() != 2)
            throw new IllegalArgumentException("Gemm inputs must be matrices");
        ValueOps.sameType(context.input(0), context.input(1));
        Tensor output = a.matmul(b).multiplyScalar(context.floating("alpha", 1));
        return context.hasInput(2)
                ? output.add(context.tensor(2).multiplyScalar(context.floating("beta", 1)))
                : output;
    }

    private static Tensor power(OnnxContext context) {
        Tensor exponent = context.tensor(1);
        if (exponent.numel() == 1 && !exponent.requiresGrad())
            return context.tensor(0).pow(exponent.item());
        throw new IllegalArgumentException("Pow currently requires a constant scalar exponent");
    }

    private static Tensor fold(
            OnnxContext context, BinaryOperator<Tensor> operation, boolean average) {
        Tensor output = context.tensor(0);
        for (int i = 1; i < context.inputCount(); i++) {
            output = operation.apply(output, context.tensor(i));
        }
        return average ? output.divideScalar(context.inputCount()) : output;
    }

    private static void matrixAndVariadic(OnnxRegistry registry) {
        registry.add("MatMul", 7, "", context -> context.one(matmul(context)));
        registry.add("Gemm", 7, "alpha beta transA transB", context -> context.one(gemm(context)));
        registry.add("Pow", 7, "", context -> context.one(power(context)));
        registry.add("Sum", 8, "", context -> context.one(fold(context, Tensor::add, false)));
        registry.add("Mean", 8, "", context -> context.one(fold(context, Tensor::add, true)));
        registry.add(
                "Max",
                8,
                "",
                context -> context.one(fold(context, ArithmeticOperators::maximum, false)));
        registry.add(
                "Min",
                8,
                "",
                context ->
                        context.one(
                                fold(context, (a, b) -> maximum(a.neg(), b.neg()).neg(), false)));
    }
}
