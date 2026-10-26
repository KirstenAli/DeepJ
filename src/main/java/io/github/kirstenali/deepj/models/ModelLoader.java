package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbit;
import io.github.kirstenali.deepj.models.orbit.DeepJOrbitConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.models.nova.DeepJNova;
import io.github.kirstenali.deepj.models.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.models.loop.DeepJLoop;
import io.github.kirstenali.deepj.models.loop.DeepJLoopConfig;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;

import java.io.IOException;
import java.nio.file.Path;

public final class ModelLoader {

    private ModelLoader() {}

    public static DecoderOnlyModel load(Path path) throws IOException {
        var layout = ModelSerializer.layout(path);
        try {
            DecoderOnlyModel model = create(layout.settings().config(), layout.parameterFactory());
            layout.requireMatches(model.parameters());
            model.load(path);
            return model;
        } catch (IllegalArgumentException failure) {
            throw new IOException("Checkpoint parameters do not match the model configuration", failure);
        }
    }

    private static DecoderOnlyModel create(TransformerConfig config, ParameterFactory parameters) throws IOException {
        if (config instanceof DeepJOriginConfig origin) return new DeepJOrigin(origin, 42L, parameters);
        if (config instanceof DeepJOrbitConfig orbit) return new DeepJOrbit(orbit, 42L, parameters);
        if (config instanceof DeepJPrismConfig prism) return new DeepJPrism(prism, 42L, parameters);
        if (config instanceof DeepJNovaConfig nova) return new DeepJNova(nova, 42L, parameters);
        if (config instanceof DeepJLoopConfig loop) return new DeepJLoop(loop, 42L, parameters);
        throw new IOException("Unsupported model configuration");
    }
}
