package io.github.kirstenali.deepj.distributed;

import java.time.Duration;
import java.util.Objects;

record TcpSettings(int rank, int worldSize, String session, int timeoutMillis) {
    static TcpSettings of(int rank, int worldSize, String session, Duration timeout) {
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(timeout, "timeout");
        long millis = timeout.toMillis();
        validateRank(rank, worldSize);
        if (session.isBlank() || session.length() > 4096 || millis < 1 || millis > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Session and finite positive timeout are required");
        }
        return new TcpSettings(rank, worldSize, session, (int) millis);
    }

    private static void validateRank(int rank, int worldSize) {
        if (worldSize < 1 || worldSize > 1024 || rank < 0 || rank >= worldSize) {
            throw new IllegalArgumentException("Invalid worker rank or count");
        }
    }
}
