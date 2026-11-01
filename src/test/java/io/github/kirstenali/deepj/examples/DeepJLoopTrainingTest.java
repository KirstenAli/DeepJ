package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.nn.transformer.ModelLoader;
import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoop;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoopConfig;
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

class DeepJLoopTrainingTest {

    @TempDir Path directory;
    private Path text, base;

    @BeforeEach
    void trainingFixture() throws Exception {
        Tensor.setBackend(new CpuBackend());
        String example = "Instruction:\nHello\nResponse:\nHello there!\n<|endoftext|>\n\n";
        text = directory.resolve("examples.txt");
        Files.writeString(text, example.repeat(100));
        base = Files.createDirectory(directory.resolve("tokenizer"));
        var bpe = new BPETrainer().train(example.repeat(10), 280,
                List.of("<BOS>", "<EOS>", "<PAD>", "<|endoftext|>"));
        BPEModelIO.save(base.resolve("tokenizer.bpe"), bpe);
    }

    @Test
    void allStagesAndResumesKeepTheLoopedArchitecture() throws Exception {
        Path previous = base;
        for (DeepJ100MStage stage : DeepJ100MStage.values()) {
            previous = runAndResume(stage, previous);
        }
        var result = assertInstanceOf(DeepJLoop.class, ModelLoader.load(previous.resolve("model-final.dj")));
        assertEquals(PromptFormat.INSTRUCTION_RESPONSE, result.promptFormat());
        assertEquals(2, result.config().loops());
        assertEquals(configuration(), result.config());
    }

    private Path runAndResume(DeepJ100MStage stage, Path previous) throws Exception {
        Path output = directory.resolve(stage.directory);
        var settings = settings(stage, previous, output, null);
        StagedDecoderTrainingRunner.run(settings);
        byte[] uninterrupted = Files.readAllBytes(output.resolve("model-final.dj"));
        StagedDecoderTrainingRunner.run(settings(stage, previous, output, output.resolve("training-latest.dj")));
        assertArrayEquals(uninterrupted, Files.readAllBytes(output.resolve("model-final.dj")));
        assertEquals(-1L, Files.mismatch(previous.resolve("tokenizer.bpe"), output.resolve("tokenizer.bpe")));
        assertThrows(java.io.IOException.class, () -> StagedDecoderTrainingRunner.run(settings));
        return output;
    }

    @Test
    void defaultsReuseTheDataButNotNovaWeightsOrOutputDirectories() {
        for (DeepJ100MStage stage : DeepJ100MStage.values()) {
            var settings = DeepJLoop100MConfig.fromSystemProperties(stage);
            assertEquals(Path.of("sample_data/deepj-90m/" + stage.corpus), settings.corpus());
            assertEquals(Path.of("checkpoints/deepj-loop-100m/" + stage.directory), settings.output());
            assertEquals(stage.steps, settings.training().steps());
            assertEquals(2, settings.architecture().loops());
            assertEquals("DeepJ Loop", settings.modelName());
            assertFalse(settings.base().toString().contains("deepj-100m"));
        }
    }

    @Test
    void anExistingNovaCannotBeUsedAsTheMidTrainingInput() throws Exception {
        var nova = new io.github.kirstenali.deepj.nn.transformer.nova.DeepJNova(configuration().backbone(), 11L);
        nova.save(base.resolve("model-final.dj"));
        assertThrows(java.io.IOException.class, () -> StagedDecoderTrainingRunner.run(
                settings(DeepJ100MStage.MIDTRAIN, base, directory.resolve("wrong"), null)));
    }

    private DeepJLoop100MConfig settings(DeepJ100MStage stage, Path input, Path output, Path resume) {
        var training = new TrainingOptions(3, 1, 0.001f, 0.0001f, 1, 1, 2, 1);
        return new DeepJLoop100MConfig(stage, configuration(), training, text, input, output,
                input.resolve("model-final.dj"), resume, 512L, stage.seed);
    }

    private static DeepJLoopConfig configuration() {
        return new DeepJLoopConfig(280, 64, 8, 4, 2, 1, 1, 1, 2, 12);
    }
}
