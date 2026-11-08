package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.agent.InteractiveTerminal;
import io.github.kirstenali.deepj.agent.Qwen3AgentModel;
import io.github.kirstenali.deepj.tensor.DType;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.Callable;

/** Separate terminal launcher that opts supported models into tool calling. */
@Command(name = "deepj-agent", mixinStandardHelpOptions = true, versionProvider = DeepJCli.Version.class,
        description = "Chat with local models and approve terminal tool calls from supported models.")
public final class AgentCli implements Callable<Integer> {
    @Parameters(index = "0", paramLabel = "MODEL", description = "Model directory or checkpoint.")
    private Path path;
    @Option(names = "--workdir", description = "Directory for terminal calls (default: current directory).")
    private Path directory = Path.of(".");
    @Option(names = "--dtype", description = "Qwen3 weight storage (default: checkpoint).")
    private DType dtype;
    @Mixin private final CliOptions options = new CliOptions();
    @Mixin private final SamplingOptions sampling = new SamplingOptions();
    private final CliServices services;

    private AgentCli(CliServices services) {
        this.services = services;
        sampling.temperature = 0.7f;
        sampling.topK = 20;
    }

    public static void main(String[] args) {
        System.exit(execute(CliServices.system(), args));
    }

    static int execute(CliServices services, String... args) {
        var cli = new CommandLine(new AgentCli(services));
        cli.setOut(services.out()).setErr(services.err()).setCaseInsensitiveEnumValuesAllowed(true);
        cli.setExecutionExceptionHandler((error, command, parsed) -> {
            command.getErr().println("deepj-agent: " + CliText.label(error.toString()));
            return error instanceof IllegalArgumentException ? 2 : 1;
        });
        return cli.execute(args);
    }

    @Override
    public Integer call() throws IOException {
        SamplingSettings settings = sampling.settings();
        try (CliConsole console = services.consoles().open(services);
             CliModel model = services.models().load(path, options.backend, dtype)) {
            if (model instanceof Qwen3CliModel qwen && InteractiveTerminal.available()) {
                runAgent(console, qwen, settings);
            } else {
                console.out().println("Agent mode unavailable. Starting regular chat.");
                new ChatSession(console, model, options, services, settings).run();
            }
        }
        return 0;
    }

    private void runAgent(CliConsole console, Qwen3CliModel qwen, SamplingSettings settings) {
        var view = new CliView(console, options, services.environment());
        view.welcome(qwen);
        new AgentCliSession(console, view, directory, thinking -> new Qwen3AgentModel(qwen.chat(),
                settings.maxTokens(), settings.temperature(), settings.topK(), settings.seed(),
                settings.repetitionPenalty(), settings.contextWindow(), thinking), settings.thinking()).run();
    }
}
