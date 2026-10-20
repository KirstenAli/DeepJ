package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbit;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbitConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.models.nova.DeepJNova;
import io.github.kirstenali.deepj.models.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.persistence.ModelSettings;

import java.io.IOException;
import java.nio.file.Path;

public final class ModelLoader {

    private ModelLoader() {}

    public static DecoderOnlyModel load(Path path) throws IOException {
        ModelSettings settings = ModelSerializer.settings(path);
        if (settings == null) throw new IOException("Model has no embedded settings: " + path);
        DecoderOnlyModel model = create(settings.config());
        model.load(path);
        return model;
    }

    private static DecoderOnlyModel create(TransformerConfig config) throws IOException {
        if (config instanceof DeepJOriginConfig origin) return new DeepJOrigin(origin, 42L);
        if (config instanceof DeepJOrbitConfig orbit) return new DeepJOrbit(orbit, 42L);
        if (config instanceof DeepJPrismConfig prism) return new DeepJPrism(prism, 42L);
        if (config instanceof DeepJNovaConfig nova) return new DeepJNova(nova, 42L);
        throw new IOException("Unsupported model configuration");
    }
}
