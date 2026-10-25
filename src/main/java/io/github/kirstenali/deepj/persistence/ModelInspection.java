package io.github.kirstenali.deepj.persistence;

/** Embedded settings and storage counts, read without allocating model weights. */
public record ModelInspection(ModelSettings settings, long parameters, int tensors, long bytes) {}
