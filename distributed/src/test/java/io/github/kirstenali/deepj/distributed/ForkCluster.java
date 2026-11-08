package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.DType;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;

final class ForkCluster implements AutoCloseable {
    private final List<Process> children = new ArrayList<>();
    private final List<Path> errors = new ArrayList<>();
    private ProcessGroup group;
    private final Path checkpoints = Files.createTempDirectory("deepj-distributed-checkpoint-");

    ForkCluster(String family, DType dtype, int[] batches, int steps, String backend, String mode) throws IOException {
        this(family, dtype, batches, steps, backend, mode, Duration.ofSeconds(20));
    }

    ForkCluster(String family, DType dtype, int[] batches, int steps, String backend, String mode,
                Duration timeout) throws IOException {
        boolean tcp = Boolean.getBoolean("deepj.distributed.tcp");
        TcpProcessGroup root = tcp ? TcpProcessGroup.coordinator(new InetSocketAddress("127.0.0.1", 0),
                batches.length, "test", timeout) : null;
        try {
            for (int rank = 1; rank < batches.length; rank++) {
                launch(family, dtype, batches, steps, backend, mode, root, rank);
            }
            var settings = TcpSettings.of(0, batches.length, "test", timeout);
            group = tcp ? root : new TcpProcessGroup(settings, new PipeTransport(children, settings),
                    new InetSocketAddress(0));
        } catch (IOException failure) {
            if (root != null) root.close();
            close();
            throw failure;
        }
    }

    ProcessGroup group() { return group; }
    Path checkpoints() { return checkpoints; }

    private void launch(String family, DType dtype, int[] batches, int steps, String backend, String mode,
                        TcpProcessGroup root, int rank) throws IOException {
        Path error = Files.createTempFile("deepj-worker-", ".log");
        errors.add(error);
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        List<String> command = new ArrayList<>(List.of(java,
                "--enable-native-access=ALL-UNNAMED", "-Xmx512m", "-cp", classpath(), ForkWorker.class.getName()));
        ForkCoverage.attach(command);
        command.addAll(List.of(family, dtype.name(), "" + rank, "" + batches.length, "" + batches[rank], "" + steps,
                backend, mode, root == null ? "pipe" : "" + root.address().getPort(), checkpoints.toString()));
        children.add(new ProcessBuilder(command).redirectError(error.toFile()).start());
    }

    private static String classpath() {
        return System.getProperty("surefire.test.class.path", System.getProperty("java.class.path"));
    }

    void successful() throws Exception {
        for (int index = 0; index < children.size(); index++) {
            Process child = children.get(index);
            assertTrue(child.waitFor(20, TimeUnit.SECONDS), "Worker did not terminate");
            assertEquals(0, child.exitValue(), Files.readString(errors.get(index)));
        }
    }

    @Override public void close() {
        if (group != null) group.close();
        for (Process child : children) { child.destroyForcibly(); }
        for (Path error : errors) {
            try { Files.deleteIfExists(error); }
            catch (IOException ignored) { /* A test failure must still close every process. */ }
        }
        removeCheckpoints();
    }

    private void removeCheckpoints() {
        try (var files = Files.walk(checkpoints)) {
            for (Path file : files.sorted(java.util.Comparator.reverseOrder()).toList()) { Files.deleteIfExists(file); }
        } catch (IOException ignored) { /* Keep any earlier assertion failure. */ }
    }
}
