package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbit;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbitConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ModelLoaderTest {

    @TempDir Path directory;

    @Test
    void loadsOriginWithoutManualConfiguration() throws IOException {
        var config = new DeepJOriginConfig(16, 8, 8, 2, 1, 16);
        assertInstanceOf(DeepJOrigin.class, roundTrip(new DeepJOrigin(config, 1L)));
    }

    @Test
    void loadsOrbitWithoutManualConfiguration() throws IOException {
        var config = new DeepJOrbitConfig(16, 8, 8, 2, 1, 16);
        assertInstanceOf(DeepJOrbit.class, roundTrip(new DeepJOrbit(config, 1L)));
    }

    @Test
    void retainsInstructionPromptFormat() throws IOException {
        var config = new DeepJPrismConfig(16, 8, 8, 2, 1, 16, 4, 2);
        var original = new DeepJPrism(config, 1L);
        original.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        original.setMetadata(Map.of("steps", "22500", "source", "Frisky SFT"));
        var loaded = assertInstanceOf(DeepJPrism.class, roundTrip(original));
        assertEquals(PromptFormat.INSTRUCTION_RESPONSE, loaded.promptFormat());
        assertEquals(original.metadata(), loaded.metadata());
    }

    @Test
    void oldWeightOnlyFilesStillLoadWithExplicitConfiguration() throws IOException {
        var config = new DeepJPrismConfig(16, 8, 8, 2, 1, 16, 4, 2);
        var model = new DeepJPrism(config, 1L);
        Path file = directory.resolve("old.dj");
        ModelSerializer.save(model.parameters(), file);
        new DeepJPrism(config, 2L).load(file);
        assertThrows(IOException.class, () -> ModelLoader.load(file));
    }

    private DecoderOnlyModel roundTrip(DecoderOnlyModel original) throws IOException {
        Path file = directory.resolve("model.dj");
        original.save(file);
        DecoderOnlyModel loaded = ModelLoader.load(file);
        assertEquals(original.config(), loaded.config());
        assertArrayEquals(original.parameters().get(0).value.data,
                loaded.parameters().get(0).value.data);
        return loaded;
    }
}
