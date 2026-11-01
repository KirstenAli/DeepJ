package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.nn.transformer.TransformerConfig;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.nn.transformer.orbit.DeepJOrbitConfig;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.nn.transformer.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoopConfig;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

final class ModelSettingsIO {

    private ModelSettingsIO() {}

    static void write(DataOutputStream output, ModelSettings settings) throws IOException {
        TransformerConfig config = settings.config();
        output.writeByte(type(config));
        writeCommon(output, config);
        if (config instanceof DeepJPrismConfig prism) {
            output.writeInt(prism.qRank());
            output.writeInt(prism.kvRank());
        }
        if (config instanceof DeepJNovaConfig nova) {
            output.writeInt(nova.kvHeads());
            output.writeBoolean(nova.qkNorm());
        }
        if (config instanceof DeepJLoopConfig loop) writeLoop(output, loop);
        output.writeByte(settings.promptFormat() == PromptFormat.PLAIN ? 0 : 1);
        writeMetadata(output, settings.metadata());
    }

    private static void writeMetadata(DataOutputStream output, Map<String, String> metadata)
            throws IOException {
        output.writeInt(metadata.size());
        for (var entry : new TreeMap<>(metadata).entrySet()) {
            output.writeUTF(entry.getKey());
            output.writeUTF(entry.getValue());
        }
    }

    private static void writeLoop(DataOutputStream output, DeepJLoopConfig config) throws IOException {
        output.writeInt(config.kvHeads());
        output.writeBoolean(config.qkNorm());
        output.writeInt(config.preludeLayers());
        output.writeInt(config.coreLayers());
        output.writeInt(config.codaLayers());
        output.writeInt(config.loops());
        output.writeFloat(config.coreResidualScale());
    }

    private static int type(TransformerConfig config) {
        if (config instanceof DeepJOriginConfig) return 1;
        if (config instanceof DeepJOrbitConfig) return 2;
        if (config instanceof DeepJPrismConfig) return 3;
        if (config instanceof DeepJNovaConfig) return 4;
        if (config instanceof DeepJLoopConfig) return 5;
        throw new IllegalArgumentException("Unsupported model configuration");
    }

    private static void writeCommon(DataOutputStream output, TransformerConfig config)
            throws IOException {
        output.writeInt(config.vocabSize());
        output.writeInt(config.maxSeqLen());
        output.writeInt(config.dModel());
        output.writeInt(config.nHeads());
        output.writeInt(config.nLayers());
        output.writeInt(config.dFF());
        output.writeFloat(config.initScale());
        output.writeFloat(config.gradClipNorm());
    }

    static ModelSettings read(DataInputStream input) throws IOException {
        int type = input.readUnsignedByte();
        Common common = readCommon(input);
        try {
            TransformerConfig config = config(input, type, common);
            PromptFormat prompt = promptFormat(input.readUnsignedByte());
            return new ModelSettings(config, prompt, readMetadata(input));
        } catch (IllegalArgumentException | ArithmeticException error) {
            throw new IOException("Invalid model configuration", error);
        }
    }

    private static Map<String, String> readMetadata(DataInputStream input) throws IOException {
        int count = input.readInt();
        if (count < 0 || count > 1024) throw new IOException("Invalid metadata count: " + count);
        Map<String, String> metadata = new HashMap<>();
        for (int i = 0; i < count; i++) {
            if (metadata.putIfAbsent(input.readUTF(), input.readUTF()) != null) {
                throw new IOException("Duplicate model metadata key");
            }
        }
        return metadata;
    }

    private static Common readCommon(DataInputStream input) throws IOException {
        return new Common(input.readInt(), input.readInt(), input.readInt(), input.readInt(),
                input.readInt(), input.readInt(), input.readFloat(), input.readFloat());
    }

    private static TransformerConfig config(DataInputStream input, int type, Common values)
            throws IOException {
        return switch (type) {
            case 1 -> values.origin();
            case 2 -> values.orbit();
            case 3 -> values.prism(input.readInt(), input.readInt());
            case 4 -> values.nova(input.readInt(), input.readBoolean());
            case 5 -> values.loop(input);
            default -> throw new IOException("Unsupported model type: " + type);
        };
    }

    private static PromptFormat promptFormat(int code) throws IOException {
        return switch (code) {
            case 0 -> PromptFormat.PLAIN;
            case 1 -> PromptFormat.INSTRUCTION_RESPONSE;
            default -> throw new IOException("Unsupported prompt format: " + code);
        };
    }

    private record Common(int vocab, int sequence, int width, int heads, int layers, int feedForward,
                          float scale, float clip) {

        DeepJOriginConfig origin() {
            return new DeepJOriginConfig(vocab, sequence, width, heads, layers, feedForward, scale, clip);
        }

        DeepJOrbitConfig orbit() {
            return new DeepJOrbitConfig(vocab, sequence, width, heads, layers, feedForward, scale, clip);
        }

        DeepJPrismConfig prism(int qRank, int kvRank) {
            return new DeepJPrismConfig(vocab, sequence, width, heads, layers, feedForward,
                    qRank, kvRank, scale, clip);
        }

        DeepJNovaConfig nova(int kvHeads, boolean qkNorm) {
            return new DeepJNovaConfig(vocab, sequence, width, heads, kvHeads, layers, feedForward,
                    qkNorm, scale, clip);
        }

        DeepJLoopConfig loop(DataInputStream input) throws IOException {
            int kvHeads = input.readInt();
            boolean qkNorm = input.readBoolean();
            var config = new DeepJLoopConfig(vocab, sequence, width, heads, kvHeads,
                    input.readInt(), input.readInt(), input.readInt(), input.readInt(), feedForward,
                    qkNorm, scale, clip, input.readFloat());
            if (config.nLayers() != layers) throw new IOException("Inconsistent loop layer counts");
            return config;
        }
    }
}
