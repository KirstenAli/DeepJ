package io.github.kirstenali.deepj.tensor;

public final class MetalBackendSmokeTest {
    public static void main(String[] args) {
        TensorBackend backend = new MetalBackend();

        Tensor a = new Tensor(2, 3);
        a.data[0][0] = 1; a.data[0][1] = 2; a.data[0][2] = 3;
        a.data[1][0] = 4; a.data[1][1] = 5; a.data[1][2] = 6;

        Tensor b = new Tensor(3, 2);
        b.data[0][0] = 7;  b.data[0][1] = 8;
        b.data[1][0] = 9;  b.data[1][1] = 10;
        b.data[2][0] = 11; b.data[2][1] = 12;

        Tensor c = backend.matmul(a, b);
        backend.print(c, "C = A x B");
    }
}