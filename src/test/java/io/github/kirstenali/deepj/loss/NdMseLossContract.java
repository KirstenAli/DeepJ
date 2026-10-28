package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.DType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** The same numerical contract executes independently against each selected backend. */
public abstract class NdMseLossContract {
    @ParameterizedTest @EnumSource(DType.class)
    protected void ranksAndViews(DType dtype) { NdMseChecks.ranksAndViews(dtype); }
    @ParameterizedTest @EnumSource(DType.class)
    protected void bothInputGradients(DType dtype) { NdMseChecks.bothInputGradients(dtype); }
    @ParameterizedTest @EnumSource(DType.class)
    protected void mixedAndMatrixOrder(DType dtype) { NdMseChecks.mixedAndMatrixOrder(dtype); }
    @Test protected void specialValues() { NdMseChecks.specialValues(); }
    @Test protected void invalidInputs() { NdMseChecks.invalidInputs(); }
    @Test protected void finiteDifferences() { NdMseChecks.finiteDifferences(); }
}
