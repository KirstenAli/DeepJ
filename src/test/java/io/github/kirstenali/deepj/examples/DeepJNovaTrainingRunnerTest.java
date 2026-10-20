package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.ModelLoader;
import io.github.kirstenali.deepj.models.PromptFormat;
import io.github.kirstenali.deepj.models.nova.DeepJNova;
import io.github.kirstenali.deepj.models.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DeepJNovaTrainingRunnerTest {

    @TempDir Path directory;
    private Path corpus;
    private Path tokenizerDirectory;

    @BeforeEach
    void prepareData() throws Exception {
        Tensor.setBackend(new CpuBackend());
        corpus = directory.resolve("training.txt");
        String record = "Instruction:\nHello\nResponse:\nHello! How can I help you today?\n<|endoftext|>\n\n";
        Files.writeString(corpus, record.repeat(100));
        tokenizerDirectory = Files.createDirectory(directory.resolve("frisky"));
        var tokenizer = new BPETrainer().train(record.repeat(10), 280,
                List.of("<BOS>", "<EOS>", "<PAD>", "<|endoftext|>"));
        BPEModelIO.save(tokenizerDirectory.resolve("tokenizer.bpe"), tokenizer);
    }

    @Test
    void pretrainingResumeReproducesTheUninterruptedRun() throws Exception {
        Path output = directory.resolve("pretrain");
        var config = config(DeepJ100MStage.PRETRAIN, tokenizerDirectory, output, null);
        DeepJNovaTrainingRunner.run(config);
        byte[] expected = Files.readAllBytes(output.resolve("model-final.dj"));
        var resumed = config(DeepJ100MStage.PRETRAIN, tokenizerDirectory, output,
                output.resolve("training-latest.dj"));
        DeepJNovaTrainingRunner.run(resumed);
        assertArrayEquals(expected, Files.readAllBytes(output.resolve("model-final.dj")));
        assertEquals(-1L, Files.mismatch(tokenizerDirectory.resolve("tokenizer.bpe"),
                output.resolve("tokenizer.bpe")));
        assertThrows(java.io.IOException.class, () -> DeepJNovaTrainingRunner.run(config));
    }

    @Test
    void allThreeStagesProduceLoadableNovaModels() throws Exception {
        Path pretrain = directory.resolve("pretrain");
        Path midtrain = directory.resolve("midtrain");
        Path fineTune = directory.resolve("sft");
        DeepJNovaTrainingRunner.run(config(DeepJ100MStage.PRETRAIN, tokenizerDirectory, pretrain, null));
        DeepJNovaTrainingRunner.run(config(DeepJ100MStage.MIDTRAIN, pretrain, midtrain, null));
        DeepJNovaTrainingRunner.run(config(DeepJ100MStage.FINE_TUNE, midtrain, fineTune, null));
        assertStageResumesExactly(DeepJ100MStage.MIDTRAIN, pretrain, midtrain);
        assertStageResumesExactly(DeepJ100MStage.FINE_TUNE, midtrain, fineTune);
        var trained = assertInstanceOf(DeepJNova.class, ModelLoader.load(fineTune.resolve("model-final.dj")));
        assertEquals(PromptFormat.INSTRUCTION_RESPONSE, trained.promptFormat());
        assertEquals(architecture(), trained.config());
        assertTrue(Files.isRegularFile(fineTune.resolve("training-latest.dj")));
    }

    private void assertStageResumesExactly(DeepJ100MStage stage, Path base, Path output) throws Exception {
        byte[] expected = Files.readAllBytes(output.resolve("model-final.dj"));
        DeepJNovaTrainingRunner.run(config(stage, base, output, output.resolve("training-latest.dj")));
        assertArrayEquals(expected, Files.readAllBytes(output.resolve("model-final.dj")));
    }

    @Test
    void defaultsReuseFriskyDataAndKeepOutputsSeparate() {
        for (DeepJ100MStage stage : DeepJ100MStage.values()) {
            var config = DeepJ100MConfig.fromSystemProperties(stage);
            assertEquals(Path.of("sample_data/deepj-90m/" + stage.corpus), config.corpus());
            assertEquals(Path.of("checkpoints/deepj-100m/" + stage.directory), config.output());
            assertEquals(stage.steps, config.training().steps());
        }
    }

    private DeepJ100MConfig config(DeepJ100MStage stage, Path base, Path output, Path resume) {
        var options = new TrainingOptions(3, 1, 1e-3f, 1e-4f, 1, 1, 2, 0);
        return new DeepJ100MConfig(stage, architecture(), options, corpus, base, output,
                base.resolve("model-final.dj"), resume, 512L, stage.seed);
    }

    private static DeepJNovaConfig architecture() {
        return new DeepJNovaConfig(280, 64, 8, 4, 2, 1, 12);
    }
}
