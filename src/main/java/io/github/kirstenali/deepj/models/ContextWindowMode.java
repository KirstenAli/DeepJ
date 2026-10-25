package io.github.kirstenali.deepj.models;

/** Generation only: rolling mode preserves cached history rather than rebuilding each shifted window. */
public enum ContextWindowMode {
    REBUILD,
    ROLLING
}
