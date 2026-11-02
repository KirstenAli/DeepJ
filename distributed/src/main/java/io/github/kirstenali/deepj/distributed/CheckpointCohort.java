package io.github.kirstenali.deepj.distributed;

import java.util.UUID;

final class CheckpointCohort {
    private CheckpointCohort() {}

    static byte[] create(ProcessGroup group) {
        byte[] local = group.rank() == 0 ? identifier() : new byte[0];
        return group.gather("checkpoint.new-cohort", local).get(0);
    }

    private static byte[] identifier() {
        UUID id = UUID.randomUUID();
        return Wire.encode(output -> {
            output.writeLong(id.getMostSignificantBits());
            output.writeLong(id.getLeastSignificantBits());
        });
    }
}
