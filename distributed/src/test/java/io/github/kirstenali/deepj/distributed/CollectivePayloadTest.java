package io.github.kirstenali.deepj.distributed;

import java.io.IOException;
import java.util.List;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CollectivePayloadTest {
    @Test void reductionsRejectMalformedAndUnequalFloatPayloads() {
        assertThrows(IOException.class, () -> CollectiveValues.sum(List.of(new byte[3])));
        assertThrows(IOException.class, () -> CollectiveValues.sum(List.of(new byte[4], new byte[8])));
        assertThrows(CollectiveException.class, () -> CollectiveValues.copy(new byte[3], new float[1]));
    }

    @Test void gatheredMessagesRejectWrongCountsTruncationAndTrailingData() throws IOException {
        byte[] valid = CollectiveValues.gathered(List.of(new byte[]{1}));
        assertThrows(IOException.class, () -> CollectiveValues.unpack(valid, 2));
        assertThrows(IOException.class, () -> CollectiveValues.unpack(new byte[2], 1));
        byte[] trailing = java.util.Arrays.copyOf(valid, valid.length + 1);
        assertThrows(IOException.class, () -> CollectiveValues.unpack(trailing, 1));
        assertArrayEquals(new byte[]{1}, CollectiveValues.unpack(valid, 1).get(0));
    }
}
