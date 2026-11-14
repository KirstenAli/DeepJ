package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.GenerationSession;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class Qwen3GgufTest {
    private static final Qwen3Config CONFIG = new Qwen3Config(40, 32, 32, 2, 1, 1, 64, 32, 10000f, .02f, 1f);
    private static final int[] PROMPT = {4, 9, 12, 7, 22, 15, 30, 21, 3};
    private static final int[] CONTINUATION = {4, 9, 12, 7, 22, 15, 30, 21, 3, 33};
    @TempDir Path directory;

    @ParameterizedTest @ValueSource(booleans = {false, true})
    void matchesDenseLogitsForPrefillAndIncrementalDecoding(boolean metal) throws Exception {
        if (metal) assumeTrue(MetalBackend.isAvailable());
        var path = fixture("qwen3", false);
        assertEquals(CONFIG, Qwen3Gguf.inspect(path).settings().config());
        TensorBackend previous = Tensor.backend(), backend = metal ? new MetalBackend() : new CpuBackend();
        Tensor.setBackend(backend);
        try (var packed = Qwen3Gguf.load(path); var dense = denseSession()) {
            try (var session = packed.openGeneration(ContextWindowMode.REBUILD)) {
                assertArrayEquals(dense.apply(PROMPT).toFloatArray(), session.apply(PROMPT).toFloatArray(), 0.0003f);
                assertArrayEquals(dense.apply(CONTINUATION).toFloatArray(),
                        session.apply(CONTINUATION).toFloatArray(), 0.0003f);
            }
        } finally {
            backend.releaseResources();
            Tensor.setBackend(previous);
        }
    }

    @Test void rejectsWrongArchitectureAndOverlappingTensors() throws Exception {
        assertThrows(IOException.class, () -> Qwen3Gguf.inspect(fixture("qwen3moe", false)));
        assertThrows(IOException.class, () -> Qwen3Gguf.inspect(fixture("qwen3", true)));
    }

    private static GenerationSession denseSession() {
        var dense = new Qwen3(CONFIG, 44);
        try (var ignored = GradMode.noGrad()) {
            for (var parameter : dense.parameters()) {
                var matrix = parameter.value;
                matrix.copyFrom(Tensor.from(denseValues(parameter.name(), matrix), matrix.shape()));
            }
        }
        return dense.openGeneration();
    }

    private static float[] denseValues(String name, Tensor matrix) {
        int seed = Qwen3GgufNames.weight(name).hashCode();
        boolean norm = name.endsWith(".gamma"), transpose = name.startsWith("model.layers.") && !norm;
        float[] values = new float[matrix.numel()];
        for (int r = 0; r < matrix.matrixRows(); r++) {
            for (int c = 0; c < matrix.matrixCols(); c++) {
                int row = transpose ? c : r, col = transpose ? r : c;
                values[r * matrix.matrixCols() + c] = norm ? 1f : coefficient(seed, row, col) / 256f;
            }
        }
        return values;
    }

    private Path fixture(String architecture, boolean overlap) throws IOException {
        Path file = Files.createTempFile(directory, "qwen3-", ".gguf");
        var weights = Qwen3Weights.inventory(CONFIG);
        try (var raw = new java.io.ByteArrayOutputStream(); var out = new DataOutputStream(raw)) {
            out.writeBytes("GGUF");
            integer(out, 3);
            number(out, weights.size());
            number(out, 12);
            metadata(out, architecture);
            var entries = headers(out, weights, overlap);
            pad(out, raw.size());
            for (var entry : entries) {
                payload(out, entry);
            }
            Files.write(file, raw.toByteArray());
        }
        return file;
    }

    private static List<Entry> headers(DataOutputStream out, List<Qwen3Weights.Weight> weights, boolean overlap)
            throws IOException {
        List<Entry> result = new ArrayList<>();
        long offset = 0;
        for (var weight : weights) {
            int[] shape = weight.sourceShape();
            var entry = new Entry(Qwen3GgufNames.weight(weight.name()), shape[shape.length - 1],
                    shape.length == 1 ? 1 : shape[0], weight.norm());
            string(out, entry.name());
            integer(out, shape.length);
            number(out, entry.width());
            if (!entry.norm()) number(out, entry.rows());
            integer(out, entry.norm() ? 0 : 8);
            number(out, overlap ? 0 : offset);
            offset += entry.bytes() + padding(entry.bytes());
            result.add(entry);
        }
        return result;
    }

    private static void payload(DataOutputStream out, Entry entry) throws IOException {
        int seed = entry.name().hashCode();
        if (entry.norm()) {
            for (int i = 0; i < entry.width(); i++) { integer(out, Float.floatToRawIntBits(1f)); }
        } else {
            for (int row = 0; row < entry.rows(); row++) {
                for (int col = 0; col < entry.width(); col += 32) {
                    block(out, seed, row, col);
                }
            }
        }
        pad(out, entry.bytes());
    }

    private static void block(DataOutputStream out, int seed, int row, int col) throws IOException {
        out.writeByte(0);
        out.writeByte(0x1c);
        for (int i = col; i < col + 32; i++) { out.writeByte(coefficient(seed, row, i)); }
    }

    private static void metadata(DataOutputStream out, String architecture) throws IOException {
        string(out, "general.architecture");
        integer(out, 8);
        string(out, architecture);
        field(out, "general.file_type", 4, 7);
        field(out, "qwen3.context_length", 4, CONFIG.maxSeqLen());
        field(out, "qwen3.embedding_length", 4, CONFIG.dModel());
        field(out, "qwen3.attention.head_count", 4, CONFIG.nHeads());
        field(out, "qwen3.attention.head_count_kv", 4, CONFIG.kvHeads());
        field(out, "qwen3.block_count", 4, CONFIG.nLayers());
        field(out, "qwen3.feed_forward_length", 4, CONFIG.dFF());
        field(out, "qwen3.attention.key_length", 4, CONFIG.headDim());
        field(out, "qwen3.attention.value_length", 4, CONFIG.headDim());
        field(out, "qwen3.attention.layer_norm_rms_epsilon", 6, Float.floatToRawIntBits(1e-6f));
        field(out, "qwen3.rope.freq_base", 6, Float.floatToRawIntBits((float) CONFIG.ropeTheta()));
    }

    private static void field(DataOutputStream out, String name, int type, int bits) throws IOException {
        string(out, name);
        integer(out, type);
        integer(out, bits);
    }

    private static void string(DataOutputStream out, String value) throws IOException {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        number(out, bytes.length);
        out.write(bytes);
    }

    private static int coefficient(int seed, int row, int col) {
        return Math.floorMod(seed + row * 61 + col * 17 + row * col * 3, 39) - 19;
    }

    private static void integer(DataOutputStream out, int value) throws IOException {
        out.writeInt(Integer.reverseBytes(value));
    }
    private static void number(DataOutputStream out, long value) throws IOException {
        out.writeLong(Long.reverseBytes(value));
    }
    private static int padding(int bytes) { return (-bytes) & 31; }
    private static void pad(DataOutputStream out, int bytes) throws IOException { out.write(new byte[padding(bytes)]); }

    private record Entry(String name, int width, int rows, boolean norm) {
        int bytes() { return norm ? width * 4 : rows * (width / 32) * 34; }
    }
}
