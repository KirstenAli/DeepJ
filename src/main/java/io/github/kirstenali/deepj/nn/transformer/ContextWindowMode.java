package io.github.kirstenali.deepj.nn.transformer;

/** Generation only: rolling mode preserves cached history rather than rebuilding each shifted window. */
public enum ContextWindowMode {
    REBUILD,
    ROLLING
}
