package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class CpuRowSelectionTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(DType.class)
    void stridedMiddleAxisGathersMatchContiguousSelection(DType dtype) {
        Tensor input = values(dtype, 3, 7, 4, 16).slice(0, 1, 3).permute(0, 2, 1, 3).slice(2, 0, 7, 2);
        checkSelect(input, 0, new int[]{1, 0, 1});
        checkSelect(input, 1, new int[]{3, 0, 3, 1});
        checkSelect(input, 2, new int[]{2, 0, 2, 1});
        checkSelect(input, 2, new int[0]);
        checkSelect(input.slice(3, 1, 16, 2), 1, new int[]{3, 0, 3});
    }

    @ParameterizedTest @EnumSource(DType.class)
    void rowCopiesPreserveRawStorageBitsAndOwnTheirOutput(DType dtype) {
        Tensor storage = values(dtype, 2, 2, 3);
        putSpecialBits(storage);
        Tensor input = storage.permute(1, 0, 2), output = input.indexSelect(0, 1, 0, 1);
        int[] sources = {3, 4, 5, 9, 10, 11, 0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11};
        for (int index = 0; index < sources.length; index++) {
            assertEquals(bits(storage, sources[index]), bits(output, index));
        }
        assertNotSame(input.memory(), output.memory());
    }

    @Test void middleAxisUpdatesPreserveRepeatedIndexOrderAcrossPermutations() {
        Tensor target = values(DType.FLOAT32, 2, 3, 4, 7).permute(0, 2, 1, 3);
        Tensor updates = values(DType.FLOAT32, 2, 3, 4, 7).permute(0, 2, 1, 3);
        checkAdd(target, 1, new int[]{2, 0, 2, 1}, updates);
        target = values(DType.FLOAT32, 2, 3, 4, 7).permute(0, 2, 1, 3);
        checkAdd(target, 2, new int[]{2, 0, 2, 1}, values(DType.FLOAT32, 2, 4, 4, 7).permute(0, 2, 1, 3));
    }

    @Test void lastAxisUpdatesSupportOffsetsAndNonunitSourceAndTargetStrides() {
        Tensor target = values(DType.FLOAT32, 3, 4, 3, 15).slice(0, 1, 3).slice(3, 1, 15, 2);
        Tensor updates = values(DType.FLOAT32, 2, 4, 4, 3).permute(0, 1, 3, 2);
        checkAdd(target, -1, new int[]{6, 0, 6, 3}, updates);
    }

    @Test void cancellationKeepsLexicalAdditionOrderAndAliasesReadASnapshot() {
        Tensor target = Tensor.from(new float[]{1, 99, 2, 99, 3, 99, 4, 99}).slice(0, 0, 8, 2);
        Tensor updates = Tensor.from(new float[]{99, 1e20f, 99, -1e20f, 99, 3}).slice(0, 1, 6, 2);
        CpuNdIndexing.add(target, 0, new int[]{2, 2, 2}, updates);
        assertArrayEquals(new float[]{1, 2, 3, 4}, target.toFloatArray());
        Tensor source = target.slice(0, 0, 4);
        CpuNdIndexing.add(target, 0, new int[]{3, 0, 3, 1}, source);
        assertArrayEquals(new float[]{3, 6, 3, 8}, target.toFloatArray());
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactUpdatesIntoFloat32KeepTheirDecodingFallback(DType dtype) {
        Tensor target = values(DType.FLOAT32, 4, 3).transpose(0, 1);
        Tensor updates = values(dtype, 4, 3).transpose(0, 1);
        checkAdd(target, 0, new int[]{2, 0, 2}, updates);
    }

    @Test void zeroWidthAndEmptySelectionsLeaveTargetsUnchanged() {
        Tensor target = new Tensor(2, 0, 3).permute(1, 0, 2);
        CpuNdIndexing.add(target, 0, new int[0], new Tensor(0, 2, 3));
        assertArrayEquals(new float[0], target.toFloatArray());
        assertArrayEquals(new float[0], target.indexSelect(0, new int[0]).toFloatArray());
    }

    private static void checkSelect(Tensor input, int axis, int[] indices) {
        Tensor expected = new Tensor(input).indexSelect(axis, indices);
        assertArrayEquals(expected.toFloatArray(), input.indexSelect(axis, indices).toFloatArray());
    }

    private static void checkAdd(Tensor target, int axis, int[] indices, Tensor updates) {
        Tensor expected = new Tensor(target), denseUpdates = new Tensor(updates);
        CpuNdIndexing.add(expected, axis, indices, denseUpdates);
        CpuNdIndexing.add(target, axis, indices, updates);
        assertArrayEquals(expected.toFloatArray(), target.toFloatArray());
    }

    private static void putSpecialBits(Tensor input) {
        if (input.dtype() == DType.FLOAT32) {
            input.memory().floats()[3] = Float.intBitsToFloat(0x7fc01234);
            input.memory().floats()[9] = -0.0f;
        } else {
            input.memory().shorts()[3] = (short) 0x7fc1;
            input.memory().shorts()[9] = (short) 0x8000;
        }
    }

    private static int bits(Tensor tensor, int index) {
        return tensor.dtype() == DType.FLOAT32 ? Float.floatToRawIntBits(tensor.memory().floats()[index])
                : Short.toUnsignedInt(tensor.memory().shorts()[index]);
    }

    private static Tensor values(DType dtype, int... shape) {
        float[] values = new float[new TensorShape(shape).elements()];
        for (int index = 0; index < values.length; index++) { values[index] = (index % 43 - 21) * .0625f; }
        return Tensor.from(values, dtype, shape);
    }
}
