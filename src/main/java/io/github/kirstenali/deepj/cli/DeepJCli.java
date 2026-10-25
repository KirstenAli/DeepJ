package io.github.kirstenali.deepj.cli;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Spec;
import picocli.CommandLine.Model.CommandSpec;

/** Terminal entry point; the Java library remains usable without the CLI dependencies. */
@Command(name = "deepj", mixinStandardHelpOptions = true, versionProvider = DeepJCli.Version.class,
        description = "Train in Java. Explore in your terminal.",
        footer = "Start here: deepj chat ./frisky-90m")
public final class DeepJCli implements Runnable {
    @Mixin private CliOptions options = new CliOptions();
    @Spec private CommandSpec spec;

    public static void main(String[] args) {
        System.exit(execute(CliServices.system(), args));
    }

    static int execute(CliServices services, String... args) {
        CommandLine cli = command(services);
        try {
            return cli.execute(args);
        } finally {
            services.out().flush();
            services.err().flush();
        }
    }

    static CommandLine command(CliServices services) {
        DeepJCli root = new DeepJCli();
        CommandLine cli = new CommandLine(root);
        cli.addSubcommand("chat", new CliCommands.Chat(root.options, services));
        cli.addSubcommand("generate", new CliCommands.Generate(root.options, services));
        cli.addSubcommand("inspect", new CliCommands.Inspect(services));
        cli.addSubcommand("doctor", new CliCommands.Doctor(services));
        configure(cli, services);
        cli.setExecutionStrategy(parsed -> {
            helpColors(cli, root.options.helpAnsi(services));
            return new CommandLine.RunLast().execute(parsed);
        });
        return cli;
    }

    private static void helpColors(CommandLine cli, CommandLine.Help.Ansi ansi) {
        cli.setColorScheme(new CommandLine.Help.ColorScheme.Builder(ansi).build());
        cli.getSubcommands().values().forEach(child -> helpColors(child, ansi));
    }

    private static void configure(CommandLine cli, CliServices services) {
        cli.setOut(services.out()).setErr(services.err()).setCaseInsensitiveEnumValuesAllowed(true);
        cli.setExecutionExceptionHandler((error, command, parsed) -> failure(error, command));
        cli.getSubcommands().values().forEach(child -> configure(child, services));
    }

    private static int failure(Exception error, CommandLine command) {
        command.getErr().println("deepj: " + CliText.label(error.getMessage() == null
                ? error.getClass().getSimpleName() : error.getMessage()));
        return error instanceof IllegalArgumentException ? 2 : 1;
    }

    @Override public void run() { spec.commandLine().usage(spec.commandLine().getOut()); }

    static final class Version implements CommandLine.IVersionProvider {
        @Override public String[] getVersion() {
            String version = DeepJCli.class.getPackage().getImplementationVersion();
            return new String[]{"DeepJ " + (version == null ? "development" : version)};
        }
    }
}
