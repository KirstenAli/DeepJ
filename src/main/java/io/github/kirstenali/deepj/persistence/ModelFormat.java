package io.github.kirstenali.deepj.persistence;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

record ModelFormat(int version, int count, ModelSettings settings) {
    private static final int MAGIC = 0x444A4D44;
    static final int CURRENT = 5;

    boolean typed() { return version >= 4; }
    boolean named() { return version == CURRENT; }

    static void write(DataOutputStream output, int count, ModelSettings settings,
                       boolean typed, boolean named) throws IOException {
        output.writeInt(MAGIC);
        output.writeInt(typed ? named ? CURRENT : 4 : settings == null ? 1 : 3);
        if (typed) output.writeBoolean(settings != null);
        if (settings != null) ModelSettingsIO.write(output, settings);
        output.writeInt(count);
    }

    static ModelFormat read(DataInputStream input) throws IOException {
        if (input.readInt() != MAGIC) throw new IOException("Unsupported checkpoint format: missing DeepJ header");
        int version = input.readInt();
        boolean embedded = embedded(input, version);
        ModelSettings settings = embedded ? ModelSettingsIO.read(input) : null;
        int count = input.readInt();
        if (count < 0) throw new IOException("Invalid checkpoint parameter count");
        return new ModelFormat(version, count, settings);
    }

    private static boolean embedded(DataInputStream input, int version) throws IOException {
        if (version == 1) return false;
        if (version == 3) return true;
        if (version == 4 || version == CURRENT) return CheckpointTensorIO.readFlag(input);
        throw new IOException("Unsupported model format version: " + version);
    }
}
