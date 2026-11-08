package io.github.kirstenali.deepj.distributed;

import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import java.util.regex.Matcher;

/** Keeps worker coverage separate until Maven merges the completed test run. */
final class ForkCoverage {
    private ForkCoverage() {}

    static void attach(List<String> command) {
        String directory = System.getProperty("deepj.distributed.coverageDirectory");
        if (directory == null) return;
        ManagementFactory.getRuntimeMXBean().getInputArguments().stream()
                .filter(argument -> argument.startsWith("-javaagent:") && argument.contains("jacoco"))
                .findFirst().ifPresent(agent -> command.add(1, workerAgent(agent, Path.of(directory))));
    }

    static String workerAgent(String agent, Path directory) {
        String destination = "destfile=" + directory.resolve("worker-" + UUID.randomUUID() + ".exec");
        return agent.replaceFirst("destfile=[^,]*", Matcher.quoteReplacement(destination));
    }
}
