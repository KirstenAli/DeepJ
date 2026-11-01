package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.List;
import java.util.Objects;

/** Composable encoder/decoder, sharing the same parameter and autograd system as every DeepJ layer. */
public final class Autoencoder extends Module {
    private final Layer encoder;
    private final Layer decoder;

    public Autoencoder(Layer encoder, Layer decoder) {
        this.encoder = Objects.requireNonNull(encoder, "encoder");
        this.decoder = Objects.requireNonNull(decoder, "decoder");
    }

    public Tensor encode(Tensor input) { return encoder.forward(input); }
    public Tensor decode(Tensor encoded) { return decoder.forward(encoded); }
    @Override protected List<Layer> children() { return List.of(encoder, decoder); }

    @Override public Tensor forward(Tensor input) {
        try (var scope = TensorScope.open()) { return scope.escape(decode(encode(input))); }
    }
}
