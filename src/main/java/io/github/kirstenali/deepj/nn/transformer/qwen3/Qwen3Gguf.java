package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.persistence.ModelInspection;
import io.github.kirstenali.deepj.persistence.ModelSettings;
import io.github.kirstenali.deepj.persistence.gguf.GgufFile;
import io.github.kirstenali.deepj.tensor.PackedQ8;
import io.github.kirstenali.deepj.tensor.TensorEncoding;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/** Loads GGUF v3 Qwen3 Q8_0 inference weights. The tokenizer remains a separate asset. */
public final class Qwen3Gguf {
    private Qwen3Gguf() {}

    public static ModelInspection inspect(Path path) throws IOException {
        var file = GgufFile.inspect(path);
        Qwen3Config config = config(file);
        validate(file, config);
        var settings = new ModelSettings(config, PromptFormat.PLAIN, Map.of("name", "Qwen3 Q8_0"));
        long elements = file.tensors().values().stream().mapToLong(entry -> (long) entry.rows() * entry.width()).sum();
        return new ModelInspection(settings, elements, file.tensors().size(), file.size());
    }

    public static Qwen3Q8 load(Path path) throws IOException {
        var file = GgufFile.inspect(path);
        Qwen3Config config = config(file);
        validate(file, config);
        Map<String, PackedQ8> packed = new HashMap<>();
        try (var input = new RandomAccessFile(path.toFile(), "r")) {
            loadMatrices(file, config, input, packed);
            var model = new Qwen3Q8(config, packed);
            return restoreNorms(file, model, input);
        } catch (IOException | RuntimeException | Error failure) {
            packed.values().forEach(PackedQ8::close);
            throw failure;
        }
    }

    private static void loadMatrices(GgufFile file, Qwen3Config config, RandomAccessFile input,
                                     Map<String, PackedQ8> packed) throws IOException {
        for (var weight : Qwen3Weights.inventory(config)) {
            if (!weight.norm()) {
                loadMatrix(input, packed, weight.name(), file.tensors().get(Qwen3GgufNames.weight(weight.name())));
            }
        }
        var output = file.tensors().get("output.weight");
        if (output != null) loadMatrix(input, packed, "lm_head.weight", output);
    }

    private static void loadMatrix(RandomAccessFile input, Map<String, PackedQ8> packed, String name,
                                   GgufFile.Entry entry) throws IOException {
        input.seek(entry.offset());
        packed.put(name, PackedQ8.read(input, entry.rows(), entry.width()));
    }

    private static Qwen3Q8 restoreNorms(GgufFile file, Qwen3Q8 model, RandomAccessFile input) throws IOException {
        try {
            loadNorms(file, model, input);
            return model;
        } catch (IOException | RuntimeException | Error failure) {
            model.close();
            throw failure;
        }
    }

    private static void loadNorms(GgufFile file, Qwen3Q8 model, RandomAccessFile input) throws IOException {
        for (var parameter : model.normalizationWeights()) {
            input.seek(file.tensors().get(Qwen3GgufNames.weight(parameter.name())).offset());
            TensorEncoding.readValues(input, parameter.value, ByteOrder.LITTLE_ENDIAN);
        }
    }

    private static Qwen3Config config(GgufFile file) throws IOException {
        if (!"qwen3".equals(file.metadata().get("general.architecture"))
                || !Long.valueOf(7).equals(file.metadata().get("general.file_type"))) {
            throw new IOException("Expected a dense Qwen3 Q8_0 GGUF");
        }
        try { return dimensions(file); }
        catch (IllegalArgumentException | ArithmeticException failure) {
            throw new IOException("Invalid Qwen3 GGUF configuration", failure);
        }
    }

    private static Qwen3Config dimensions(GgufFile file) {
        var metadata = file.metadata();
        int key = integer(metadata, "attention.key_length"), value = integer(metadata, "attention.value_length");
        requireDefaultRope(metadata, key);
        if (key != value || !Float.valueOf(1e-6f).equals(metadata.get("qwen3.attention.layer_norm_rms_epsilon"))) {
            throw new IllegalArgumentException("Unsupported Qwen3 attention geometry");
        }
        Object base = metadata.get("qwen3.rope.freq_base");
        if (!(base instanceof Float frequency)) throw new IllegalArgumentException("Missing Qwen3 RoPE base");
        return new Qwen3Config(vocabulary(file), integer(metadata, "context_length"),
                integer(metadata, "embedding_length"), integer(metadata, "attention.head_count"),
                integer(metadata, "attention.head_count_kv"), integer(metadata, "block_count"),
                integer(metadata, "feed_forward_length"), key, frequency, 0.02f, 1);
    }

    private static void requireDefaultRope(Map<String, Object> metadata, int headWidth) {
        Object scaling = metadata.getOrDefault("qwen3.rope.scaling.type", "none");
        if (!"none".equals(scaling) || !Long.valueOf(headWidth).equals(
                metadata.getOrDefault("qwen3.rope.dimension_count", (long) headWidth))
                || !Long.valueOf(0).equals(metadata.getOrDefault("qwen3.attention.sliding_window", 0L))) {
            throw new IllegalArgumentException("Unsupported Qwen3 RoPE scaling or sliding attention");
        }
    }

    private static int vocabulary(GgufFile file) {
        var embedding = file.tensors().get("token_embd.weight");
        if (embedding == null) throw new IllegalArgumentException("Missing Qwen3 embeddings");
        return embedding.rows();
    }

    private static int integer(Map<String, Object> fields, String name) {
        Object value = fields.get("qwen3." + name);
        if (value instanceof Long number) return Math.toIntExact(number);
        throw new IllegalArgumentException("Missing Qwen3 metadata: " + name);
    }

    private static void validate(GgufFile file, Qwen3Config config) throws IOException {
        var inventory = Qwen3Weights.inventory(config);
        boolean hasOutput = file.tensors().containsKey("output.weight");
        if (file.tensors().size() != inventory.size() + (hasOutput ? 1 : 0)) {
            throw new IOException("Qwen3 GGUF tensor count does not match its architecture");
        }
        for (var weight : inventory) {
            int rows = weight.transpose() ? weight.columns() : weight.rows();
            int width = weight.transpose() ? weight.rows() : weight.columns();
            int type = weight.norm() ? GgufFile.F32 : GgufFile.Q8_0;
            require(file, Qwen3GgufNames.weight(weight.name()), rows, width, type);
        }
        if (hasOutput) require(file, "output.weight", config.vocabSize(), config.dModel(), GgufFile.Q8_0);
    }

    private static void require(GgufFile file, String name, int rows, int width, int type) throws IOException {
        var entry = file.tensors().get(name);
        if (entry == null || entry.rows() != rows || entry.width() != width || entry.type() != type) {
            throw new IOException("Missing or incompatible GGUF tensor: " + name);
        }
    }
}
