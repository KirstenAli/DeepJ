package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

final class LayoutTransformsTest {
    @Test
    void permutationAndSelectionPreservePhysicalAddresses() {
        var original = TensorLayout.dense(new TensorShape(2, 3, 4));
        var permuted = LayoutTransforms.permute(original, -1, 0, 1);
        assertEquals(new TensorShape(4, 2, 3), permuted.shape());
        assertArrayEquals(new long[]{1, 12, 4}, permuted.strides());
        assertEquals(23, permuted.address(3, 1, 2));
        assertFalse(permuted.contiguous());
        assertTrue(permuted.nonOverlapping());
        var selected = LayoutTransforms.select(original, 1, -2);
        assertEquals(new TensorShape(2, 4), selected.shape());
        assertEquals(4, selected.offset());
        assertEquals(19, selected.address(1, 3));
    }

    @Test
    void slicesAndOffsetViewsStayInsideTheSameAllocation() {
        var original = TensorLayout.dense(new TensorShape(2, 3, 4));
        var slice = LayoutTransforms.slice(original, 1, 1, 3, 1);
        assertEquals(new TensorShape(2, 2, 4), slice.shape());
        assertEquals(4, slice.addressAt(0));
        assertEquals(16, slice.addressAt(8));
        assertEquals(23, slice.addressAt(15));
        var row = LayoutTransforms.select(LayoutTransforms.select(original, 1, 1), 0, 1);
        var view = LayoutTransforms.view(row, 2, 2);
        assertEquals(16, view.offset());
        assertEquals(24, view.storageElements());
        assertEquals(19, view.address(1, 1));
    }

    @Test
    void steppedAndEmptySlicesHavePredictableLayouts() {
        var original = TensorLayout.dense(new TensorShape(8));
        var stepped = LayoutTransforms.slice(original, 0, 1, 8, 2);
        assertEquals(new TensorShape(4), stepped.shape());
        assertEquals(7, stepped.addressAt(3));
        assertFalse(stepped.contiguous());
        assertTrue(stepped.nonOverlapping());
        var empty = LayoutTransforms.slice(original, 0, 8, 8, 3);
        assertEquals(0, empty.shape().elements());
        assertTrue(empty.contiguous());
        assertThrows(IllegalArgumentException.class, () -> LayoutTransforms.view(stepped, 2, 2));
    }

    @Test
    void expansionUsesZeroStridesWithoutCreatingWritableOverlaps() {
        var row = TensorLayout.dense(new TensorShape(1, 3));
        var expanded = LayoutTransforms.expand(row, 2, 4, 3);
        assertArrayEquals(new long[]{0, 0, 1}, expanded.strides());
        assertEquals(2, expanded.address(1, 3, 2));
        assertThrows(IllegalStateException.class, expanded::requireWritable);
        var empty = LayoutTransforms.expand(TensorLayout.dense(new TensorShape(0, 3)), 2, 0, 3);
        assertEquals(0, empty.shape().elements());
        assertDoesNotThrow(empty::requireWritable);
    }

    @Test
    void invalidTransformsFailBeforeAnyStorageIsChanged() {
        var layout = TensorLayout.dense(new TensorShape(2, 3));
        assertThrows(IllegalArgumentException.class, () -> LayoutTransforms.permute(layout, 0));
        assertThrows(IllegalArgumentException.class, () -> LayoutTransforms.permute(layout, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> LayoutTransforms.permute(layout, 0, 2));
        assertThrows(IllegalArgumentException.class, () -> LayoutTransforms.expand(layout, 3));
        assertThrows(IllegalArgumentException.class, () -> LayoutTransforms.expand(layout, 2, 4));
        assertThrows(IllegalArgumentException.class, () -> LayoutTransforms.slice(layout, 0, 0, 2, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> LayoutTransforms.slice(layout, 0, -1, 2, 1));
        assertThrows(IndexOutOfBoundsException.class, () -> LayoutTransforms.slice(layout, 0, 2, 1, 1));
        assertThrows(IndexOutOfBoundsException.class, () -> LayoutTransforms.slice(layout, 0, 0, 3, 1));
    }
}
