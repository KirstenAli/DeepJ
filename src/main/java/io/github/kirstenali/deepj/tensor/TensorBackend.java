package io.github.kirstenali.deepj.tensor;

import java.util.Random;

public interface TensorBackend {
    // factories (static-style)
    Tensor zeros(int rows, int cols);
    Tensor ones(int rows, int cols);
    Tensor random(int rows, int cols, Random rand);
    Tensor causalMask(int size);

    Tensor unflattenToTensor(double[] flat, int rows, int cols);
    double[] flattenTensor(Tensor t);

    // ops (instance-style)
    Tensor matmul(Tensor a, Tensor b);

    Tensor add(Tensor a, Tensor b);
    Tensor subtract(Tensor a, Tensor b);
    Tensor multiply(Tensor a, Tensor b);
    Tensor divide(Tensor a, Tensor b);

    Tensor addRowVector(Tensor a, Tensor rowVector);

    Tensor sumRows(Tensor a);

    Tensor clamp(Tensor a, double min, double max);

    Tensor transpose(Tensor a);

    Tensor multiplyScalar(Tensor a, double scalar);
    Tensor addScalar(Tensor a, double scalar);
    Tensor divideScalar(Tensor a, double scalar);

    Tensor meanAlongRows(Tensor a);
    Tensor varianceAlongRows(Tensor a);

    Tensor sumAlongRows(Tensor a);
    Tensor sumAlongCols(Tensor a);

    Tensor addBroadcastCols(Tensor a, Tensor colVector);
    Tensor divideBroadcastCols(Tensor a, Tensor colVector);
    Tensor subtractBroadcastCols(Tensor a, Tensor colVector);
    Tensor multiplyBroadcastCols(Tensor a, Tensor colVector);

    Tensor addBroadcastRows(Tensor a, Tensor rowVector);
    Tensor multiplyBroadcastRows(Tensor a, Tensor rowVector);

    Tensor sqrt(Tensor a);
    Tensor pow(Tensor a, double exponent);

    double sum(Tensor a);
    double sumAbs(Tensor a);

    void print(Tensor t, String label);
}