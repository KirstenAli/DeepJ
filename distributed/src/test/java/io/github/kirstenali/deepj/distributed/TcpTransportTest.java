package io.github.kirstenali.deepj.distributed;

import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

@Timeout(15)
@EnabledIfSystemProperty(named = "deepj.distributed.tcp", matches = "true")
class TcpTransportTest {
    @ParameterizedTest @ValueSource(strings = {"wrong-session", "wrong-world"})
    void rendezvousRejectsMismatchedJobs(String mode) throws Exception {
        try (var root = TcpProcessGroup.coordinator(new InetSocketAddress("127.0.0.1", 0),
                2, "test", Duration.ofSeconds(3))) {
            var worker = CompletableFuture.runAsync(() -> rejectedWorker(root.address(), mode));
            assertThrows(CollectiveException.class, () -> root.barrier("hello"));
            worker.get(5, TimeUnit.SECONDS);
        }
    }

    private static void rejectedWorker(InetSocketAddress address, String mode) {
        String session = mode.equals("wrong-session") ? "other" : "test";
        int workers = mode.equals("wrong-world") ? 3 : 2;
        assertThrows(Exception.class, () -> {
            try (var group = TcpProcessGroup.connect(address, 1, workers, session, Duration.ofSeconds(3))) {
                group.barrier("hello");
            }
        });
    }
}
