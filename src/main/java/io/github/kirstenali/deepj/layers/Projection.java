package io.github.kirstenali.deepj.layers;

/**
 * Marker interface for linear projection layers (fully-connected, no activation).
 *
 * <p>Narrows {@link Layer} to layers that map {@code [n × dIn] → [n × dOut]} via a
 * learnable weight matrix (and optional bias).  Used as the field type for the
 * LM-head slot in {@link io.github.kirstenali.deepj.models.DecoderOnlyModel}
 * so the constructor rejects anything that is not a projection.
 */
public interface Projection extends Layer {
}

