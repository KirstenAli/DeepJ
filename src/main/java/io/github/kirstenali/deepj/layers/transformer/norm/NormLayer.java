package io.github.kirstenali.deepj.layers.transformer.norm;

import io.github.kirstenali.deepj.layers.Layer;

/**
 * Marker interface for row-wise normalisation layers (LayerNorm, RMSNorm, …).
 *
 * <p>Narrows {@link Layer} to layers whose sole job is to normalise the feature
 * dimension of a {@code [seqLen × dim]} tensor.  Used as the field type for the
 * final-norm slot in {@link io.github.kirstenali.deepj.models.DecoderOnlyModel}
 * so the constructor rejects anything that is not a normalisation layer.
 */
public interface NormLayer extends Layer {
}

