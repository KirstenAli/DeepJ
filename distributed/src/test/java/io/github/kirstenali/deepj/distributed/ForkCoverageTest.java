package io.github.kirstenali.deepj.distributed;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class ForkCoverageTest {
    @TempDir Path directory;

    @Test void workerFilesAreDistinctAndPreserveAgentSettingsWithSpecialPathCharacters() {
        String agent = "-javaagent:/tmp/jacoco.jar=destfile=/tmp/parent.exec,append=false,includes=example.*";
        Path target = directory.resolve("space and $dollars");
        String first = ForkCoverage.workerAgent(agent, target);
        String second = ForkCoverage.workerAgent(agent, target);
        assertNotEquals(first, second);
        assertTrue(first.startsWith("-javaagent:/tmp/jacoco.jar=destfile=" + target.resolve("worker-")));
        assertTrue(first.endsWith(".exec,append=false,includes=example.*"));
    }
}
