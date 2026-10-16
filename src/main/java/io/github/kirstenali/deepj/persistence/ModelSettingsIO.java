package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.models.PromptFormat;
import io.github.kirstenali.deepj.models.TransformerConfig;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbitConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

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
        output.writeByte(settings.promptFormat() == PromptFormat.PLAIN ? 0 : 1);
    }

    private static int type(TransformerConfig config) {
        if (config instanceof DeepJOriginConfig) return 1;
        if (config instanceof DeepJOrbitConfig) return 2;
        if (config instanceof DeepJPrismConfig) return 3;
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
            return new ModelSettings(config, promptFormat(input.readUnsignedByte()));
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid model configuration", error);
        }
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
    }
}
