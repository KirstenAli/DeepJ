package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.TrainingResult;

public final class FineTuneDeepSeekKnowledge {

    private FineTuneDeepSeekKnowledge() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        System.out.println("Backend: " + Tensor.backend().getClass().getSimpleName());
        run(KnowledgeFineTuningConfig.fromSystemProperties());
    }

    static TrainingResult run(KnowledgeFineTuningConfig config) throws Exception {
        return ResponseOnlyFineTuner.run(config.responseConfig());
    }
}
