package io.github.kirstenali.deepj.persistence.qwen3;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3JsonTest {
    @Test void stringsIntegersDecimalsAndNestedValuesRoundTripWithoutLosingLongPrecision() {
        Map<String, Object> values = Map.of("text", "\"\\\n\t😀 café", "long", Long.MAX_VALUE,
                "values", List.of(true, 1.5, Map.of("a", -2)));
        var decoded = Qwen3Json.object(Qwen3Json.parse(Qwen3Json.stringify(values)));
        assertEquals(values.get("text"), decoded.get("text"));
        assertEquals(Long.MAX_VALUE, Qwen3Json.integer(decoded.get("long")));
        assertNull(Qwen3Json.parse("null"));
        assertEquals(1.25e-10, Qwen3Json.number(Qwen3Json.parse("1.25e-10")));
    }

    @ParameterizedTest
    @ValueSource(strings = {"{\"a\":null,\"a\":1}", "[1,]", "{\"a\":1,}", "01", "- 1", "1. 2",
            "1e 2", "1e999", "9223372036854775808", "true false", "\"a\n\"", "\"\\x\"", "[", ""})
    void malformedOrAmbiguousJsonIsRejected(String json) {
        assertThrows(IllegalArgumentException.class, () -> Qwen3Json.parse(json));
    }

    @Test void nestingAndNonFiniteOutputAreBounded() {
        assertThrows(IllegalArgumentException.class, () -> Qwen3Json.parse("[".repeat(65) + "]".repeat(65)));
        assertThrows(IllegalArgumentException.class, () -> Qwen3Json.stringify(Float.NaN));
    }
}
