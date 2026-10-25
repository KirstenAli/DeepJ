package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.persistence.ModelInspection;
import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Parameters;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.Callable;

final class CliCommands {
    private CliCommands() {}

    @Command(name = "chat", mixinStandardHelpOptions = true, description = "Open a streaming terminal chat.")
    static final class Chat implements Callable<Integer> {
        @Parameters(index = "0", paramLabel = "MODEL", description = "Model directory or .dj checkpoint.")
        private Path path;
        @Mixin private SamplingOptions sampling = new SamplingOptions();
        private final CliOptions options;
        private final CliServices services;

        Chat(CliOptions options, CliServices services) {
            this.options = options;
            this.services = services;
        }

        @Override public Integer call() throws IOException {
            SamplingSettings settings = sampling.settings();
            try (CliConsole console = services.consoles().open(services);
                 CliModel model = services.models().load(path, options.backend)) {
                new ChatSession(console, model, options, services, settings).run();
            }
            return 0;
        }
    }

    @Command(name = "generate", mixinStandardHelpOptions = true,
            description = "Stream one answer to stdout. Safe for pipes and scripts.")
    static final class Generate implements Callable<Integer> {
        @Parameters(index = "0", paramLabel = "MODEL", description = "Model directory or .dj checkpoint.")
        private Path path;
        @Mixin private SamplingOptions sampling = new SamplingOptions();
        @Mixin private PromptInput input = new PromptInput();
        private final CliOptions options;
        private final CliServices services;

        Generate(CliOptions options, CliServices services) {
            this.options = options;
            this.services = services;
        }

        @Override public Integer call() throws IOException {
            SamplingSettings settings = sampling.settings();
            String prompt = input.read(services);
            try (CliModel model = services.models().load(path, options.backend)) {
                StreamedText output = new StreamedText(services.out());
                output.complete(model.stream(prompt, settings, output::update));
                if (!output.endsWithNewline()) services.out().println();
            }
            return 0;
        }
    }

    @Command(name = "inspect", mixinStandardHelpOptions = true,
            description = "Inspect embedded settings without loading model weights.")
    static final class Inspect implements Callable<Integer> {
        @Parameters(index = "0", paramLabel = "MODEL", description = "Model directory or .dj checkpoint.")
        private Path path;
        private final CliServices services;

        Inspect(CliServices services) { this.services = services; }

        @Override public Integer call() throws IOException {
            ModelInspection info = services.inspector().inspect(path);
            var out = services.out();
            out.printf("Architecture    %s%n", CliModel.architecture(info));
            out.printf("Parameters      %,d%nContext         %,d tokens%n",
                    info.parameters(), info.settings().config().maxSeqLen());
            out.printf("Vocabulary      %,d%nTensors         %,d%n",
                    info.settings().config().vocabSize(), info.tensors());
            out.printf("Checkpoint      %,.2f MiB%n", info.bytes() / (1024.0 * 1024.0));
            out.printf("Prompt format   %s%n", info.settings().promptFormat());
            return 0;
        }
    }

    @Command(name = "doctor", mixinStandardHelpOptions = true,
            description = "Check Java and backend availability without loading a model.")
    static final class Doctor implements Callable<Integer> {
        private final CliServices services;

        Doctor(CliServices services) { this.services = services; }

        @Override public Integer call() {
            boolean metal = services.metalAvailable().getAsBoolean();
            services.out().println("Java            " + System.getProperty("java.version"));
            services.out().println("CPU             available");
            services.out().println("Metal           " + (metal ? "available" : "unavailable; CPU fallback"));
            services.out().println("Auto backend    " + (metal ? "Metal" : "CPU"));
            return 0;
        }
    }
}
