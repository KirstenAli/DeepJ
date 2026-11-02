package io.github.kirstenali.deepj.distributed;

import java.io.IOException;
import java.util.Arrays;

final class RankAgreement {
    private RankAgreement() {}

    static void same(ProcessGroup group, String key, byte[] bytes) {
        for (byte[] value : group.gather(key, bytes)) {
            if (!Arrays.equals(bytes, value)) fail(group, key + " differs between workers");
        }
    }

    static void success(ProcessGroup group, String key, Exception failure) {
        String message = failure == null ? "" : failure.getClass().getSimpleName() + ": " + failure.getMessage();
        byte[] local = Wire.encode(output -> output.writeUTF(message.substring(0, Math.min(message.length(), 4000))));
        for (byte[] value : group.gather(key, local)) {
            try (var input = Wire.input(value)) {
                String error = input.readUTF();
                if (!error.isEmpty()) fail(group, key + ": " + error);
            } catch (IOException error) { fail(group, "Invalid worker status"); }
        }
    }

    static void fail(ProcessGroup group, String message) {
        group.abort(message);
        throw new CollectiveException(message);
    }
}
