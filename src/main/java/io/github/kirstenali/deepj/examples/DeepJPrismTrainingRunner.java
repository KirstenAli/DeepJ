package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.RandomAccessTextDataset;
import io.github.kirstenali.deepj.data.SequentialTextDataset;
import io.github.kirstenali.deepj.data.StatefulTrainingDataset;
import io.github.kirstenali.deepj.data.TextFileRange;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrism;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModel;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import io.github.kirstenali.deepj.training.TrainingResult;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class DeepJPrismTrainingRunner {

    private static final List<String> SPECIAL_TOKENS =
            List.of("<BOS>", "<EOS>", "<PAD>", "<|endoftext|>");
    private static final String TOKENIZER_FILE = "tokenizer.bpe";

    private DeepJPrismTrainingRunner() {}

    static TrainingResult run(DeepJPrismTrainingConfig runConfig) throws Exception {
        return run(runConfig, DatasetOrder.random());
    }

    static TrainingResult runSequential(DeepJPrismTrainingConfig runConfig)
            throws Exception {
        return run(runConfig, DatasetOrder.sequential(null));
    }

    static TrainingResult runSequential(DeepJPrismTrainingConfig runConfig,
                                        TextFileRange range) throws Exception {
        return run(runConfig, DatasetOrder.sequential(range));
    }

    private static TrainingResult run(DeepJPrismTrainingConfig runConfig,
                                      DatasetOrder order) throws Exception {
        Path output = runConfig.files().outputDirectory();
        validateCorpus(runConfig.files().corpus());
        Files.createDirectories(output);
        BPETokenizer tokenizer = prepareTokenizer(runConfig);
        DeepJPrismConfig modelConfig = runConfig.modelConfig(tokenizer.vocabSize());
        DeepJPrism model = new DeepJPrism(modelConfig, runConfig.seed());
        DecoderTrainingSession.ensureCheckpointSpace(output, model);
        model.setMetadata(metadata(runConfig, order));
        try (StatefulTrainingDataset dataset = dataset(runConfig, tokenizer, order)) {
            return train(model, dataset, runConfig);
        }
    }

    private static StatefulTrainingDataset dataset(DeepJPrismTrainingConfig config,
                                                    BPETokenizer tokenizer,
                                                    DatasetOrder order) throws IOException {
        int length = config.architecture().sequenceLength();
        if (order.sequential()) return sequentialDataset(config, tokenizer, length, order.range());
        return new RandomAccessTextDataset(config.files().corpus(), tokenizer, length, config.seed());
    }

    private static StatefulTrainingDataset sequentialDataset(
            DeepJPrismTrainingConfig config, BPETokenizer tokenizer,
            int length, TextFileRange range) throws IOException {
        if (range == null) return new SequentialTextDataset(config.files().corpus(), tokenizer, length);
        return new SequentialTextDataset(config.files().corpus(), tokenizer, length, range);
    }

    static BPETokenizer prepareTokenizer(DeepJPrismTrainingConfig config)
            throws IOException {
        Path path = config.files().outputDirectory().resolve(TOKENIZER_FILE);
        Files.createDirectories(path.getParent());
        BPEModel model = Files.isRegularFile(path) ? BPEModelIO.load(path) : trainTokenizer(config, path);
        validateTokenizer(model, config.tokenizer().vocabSize());
        return new BPETokenizer(model);
    }

    private static BPEModel trainTokenizer(DeepJPrismTrainingConfig config, Path path)
            throws IOException {
        System.out.printf("Training BPE tokenizer: vocab=%d sample=%d MiB%n",
                config.tokenizer().vocabSize(), config.tokenizer().sampleMiB());
        BPEModel model = new BPETrainer().trainFromFile(config.files().corpus(),
                config.tokenizer().vocabSize(), SPECIAL_TOKENS, config.tokenizer().sampleChars());
        BPEModelIO.save(path, model);
        return model;
    }

    private static void validateTokenizer(BPEModel model, int expectedVocabSize) {
        if (model.vocabSize() != expectedVocabSize) {
            throw new IllegalStateException("Tokenizer vocabulary does not match deepj.vocabSize");
        }
        for (String token : SPECIAL_TOKENS) {
            if (!model.specialTokenToId().containsKey(token)) {
                throw new IllegalStateException("Tokenizer is missing special token " + token);
            }
        }
    }

    private static TrainingResult train(DeepJPrism model, StatefulTrainingDataset dataset,
                                        DeepJPrismTrainingConfig config) throws IOException {
        return DecoderTrainingSession.run(model, dataset, config.training(),
                config.files().outputDirectory(), config.files().resumeCheckpoint());
    }

    private static void validateCorpus(Path corpus) {
        if (!Files.isRegularFile(corpus)) {
            throw new IllegalArgumentException("Training corpus not found: " + corpus);
        }
    }

    private static Map<String, String> metadata(DeepJPrismTrainingConfig config,
                                                DatasetOrder order) {
        Map<String, String> values = new HashMap<>();
        addFileMetadata(values, config);
        addModelMetadata(values, config);
        TrainingMetadata.add(values, config.training());
        values.put("datasetOrder", order.sequential() ? "sequential" : "random");
        addRangeMetadata(values, order.range());
        return values;
    }

    private static void addRangeMetadata(Map<String, String> target, TextFileRange range) {
        if (range == null) return;
        target.put("corpusStartByte", Long.toString(range.startInclusive()));
        target.put("corpusEndByte", Long.toString(range.endExclusive()));
    }

    private static void addFileMetadata(Map<String, String> target,
                                        DeepJPrismTrainingConfig config) {
        target.put("corpus", config.files().corpus().toAbsolutePath().toString());
        target.put("output", config.files().outputDirectory().toAbsolutePath().toString());
        target.put("seed", Long.toString(config.seed()));
    }

    private static void addModelMetadata(Map<String, String> target,
                                         DeepJPrismTrainingConfig config) {
        DeepJPrismTrainingConfig.Architecture model = config.architecture();
        target.put("vocabSize", Integer.toString(config.tokenizer().vocabSize()));
        target.put("sequenceLength", Integer.toString(model.sequenceLength()));
        target.put("dModel", Integer.toString(model.dModel()));
        target.put("heads", Integer.toString(model.heads()));
        target.put("layers", Integer.toString(model.layers()));
        target.put("dFF", Integer.toString(model.dFF()));
        target.put("qRank", Integer.toString(model.qRank()));
        target.put("kvRank", Integer.toString(model.kvRank()));
        target.put("initScale", Float.toString(model.initScale()));
        target.put("gradClipNorm", Float.toString(model.gradClipNorm()));
        target.put("tokenizerSampleMiB", Integer.toString(config.tokenizer().sampleMiB()));
    }

    private record DatasetOrder(boolean sequential, TextFileRange range) {

        private static DatasetOrder random() {
            return new DatasetOrder(false, null);
        }

        private static DatasetOrder sequential(TextFileRange range) {
            return new DatasetOrder(true, range);
        }
    }
}
