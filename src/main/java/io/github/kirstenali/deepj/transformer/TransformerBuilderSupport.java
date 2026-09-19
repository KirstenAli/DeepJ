package io.github.kirstenali.deepj.transformer;

final class TransformerBuilderSupport {

    private TransformerBuilderSupport() {}

    static void validateCommon(int dModel, int nHeads, int dFF, int nLayers) {
        if (dModel <= 0)  throw new IllegalArgumentException("dModel must be > 0");
        if (nHeads <= 0)  throw new IllegalArgumentException("nHeads must be > 0");
        if (dFF <= 0)     throw new IllegalArgumentException("dFF must be > 0");
        if (nLayers <= 0) throw new IllegalArgumentException("nLayers must be > 0");
    }
}
