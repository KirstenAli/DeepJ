package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class GraphLayoutsTest {
    @Test
    void descriptorsSeparatePhysicalStorageFromLogicalAxesAndOffsets() {
        Tensor source = new Tensor(2, 3, 4);
        Tensor view = Tensor.aliasStorage(source, LayoutTransforms.select(source.layout(), 0, 1));
        Tensor transposed = Tensor.aliasStorage(view, LayoutTransforms.permute(view.layout(), 1, 0));
        var layouts = new GraphLayouts();
        assertEquals(0, layouts.add(transposed));
        assertArrayEquals(new long[]{2, 12, 24, 12, 4, 3, 1, 4}, layouts.snapshot());
        assertEquals(0, layouts.add(Tensor.aliasStorage(transposed)));
        assertEquals(8, layouts.add(source));
        assertEquals(18, layouts.snapshot().length);
    }

    @Test
    void snapshotsAreImmutableAndClearingDropsOffsetsAndUsedWords() {
        var layouts = new GraphLayouts();
        assertTrue(layouts.isEmpty());
        Tensor source = new Tensor();
        layouts.add(source);
        long[] snapshot = layouts.snapshot();
        snapshot[1] = 99;
        assertArrayEquals(new long[]{0, 1, 1, 0}, layouts.snapshot());
        layouts.clear();
        assertTrue(layouts.isEmpty());
        assertArrayEquals(new long[0], layouts.snapshot());
        assertEquals(0, layouts.add(source));
    }

    @Test
    void descriptorsHaveNoInlineArgumentRankCapAndPreserve64BitStrides() {
        int[] dimensions = new int[3000];
        Arrays.fill(dimensions, 1);
        Tensor source = new Tensor(dimensions);
        long[] strides = source.strides();
        strides[0] = Long.MAX_VALUE;
        Tensor view = Tensor.aliasStorage(source, new TensorLayout(source.layout().shape(), strides, 0, 1));
        var layouts = new GraphLayouts();
        layouts.add(view);
        long[] words = layouts.snapshot();
        assertEquals(6004, words.length);
        assertEquals(Long.MAX_VALUE, words[3004]);
        assertTrue(words.length * Long.BYTES > 4096);
        layouts.clear();
        assertEquals(0, layouts.add(new Tensor()));
        assertEquals(4, layouts.snapshot().length);
    }
}
