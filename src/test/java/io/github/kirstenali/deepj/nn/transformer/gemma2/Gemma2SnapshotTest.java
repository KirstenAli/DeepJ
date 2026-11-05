package io.github.kirstenali.deepj.nn.transformer.gemma2;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Gemma2SnapshotTest {
    @Test void checkpointRoutingAcceptsGemmaFilesAndLeavesDjBundlesToTheirExistingLoader() throws Exception {
        Path folder = Path.of("src/test/resources/gemma2");
        assertTrue(Gemma2Snapshot.matches(folder));
        assertTrue(Gemma2Snapshot.matches(folder.resolve("config.json")));
        assertTrue(Gemma2Snapshot.matches(folder.resolve("model.safetensors")));
        assertFalse(Gemma2Snapshot.matches(folder.resolve("model.dj")));
        assertFalse(Gemma2Snapshot.matches(folder.resolve("model.bin")));
    }
}
