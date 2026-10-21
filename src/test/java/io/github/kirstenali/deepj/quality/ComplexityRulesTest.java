package io.github.kirstenali.deepj.quality;

import net.sourceforge.pmd.PMDConfiguration;
import net.sourceforge.pmd.PmdAnalysis;
import net.sourceforge.pmd.lang.document.FileId;
import net.sourceforge.pmd.reporting.Report;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ComplexityRulesTest {

    private static final String NESTING = "ControlFlowNesting";
    private static final String COGNITIVE = "CognitiveComplexity";
    private static final String THREE_LEVELS = "if (x > 0) { for (int i = 0; i < x; i++) { while (x > 0) { %s } } }";

    @ParameterizedTest
    @MethodSource("fourthLevelStatements")
    void mixedControlFlowRejectsTheFourthLevel(String statement) {
        assertRule(method(THREE_LEVELS.formatted(statement)), NESTING, true);
    }

    private static Stream<String> fourthLevelStatements() {
        return Stream.of("if (x > 1) { x--; }", "for (int j = 0; j < x; j++) { x--; }",
                "for (int value : new int[]{1}) { x += value; }", "while (x > 1) { x--; }",
                "do { x--; } while (x > 1);", "switch (x) { default: x--; }",
                "x = switch (x) { default -> 1; };", "try { x--; } catch (RuntimeException e) { x++; }");
    }

    @Test
    void threeMixedLevelsAreAllowed() {
        assertRule(method(THREE_LEVELS.formatted("x--;")), NESTING, false);
    }

    @Test
    void elseIfChainsDoNotCountAsExtraNesting() {
        String branches = "if (x == 0) { x++; } " + "else if (x == 1) { x--; } ".repeat(5);
        assertRule(method(branches), NESTING, false);
    }

    @Test
    void nestedThenBranchStillCountsWhenAnElseExists() {
        String inner = "if (x > 2) if (x > 3) if (x > 4) { x--; } else { x++; }";
        assertRule(method("if (x > 1) { " + inner + " }"), NESTING, true);
    }

    @ParameterizedTest
    @ValueSource(ints = {15, 16})
    void cognitiveComplexityAllowsFifteenButRejectsSixteen(int score) {
        assertRule(method("if (x > 0) { x--; } ".repeat(score)), COGNITIVE, score > 15);
    }

    @Test
    void constructorsAreChecked() {
        String body = THREE_LEVELS.formatted("if (x > 1) { x--; }");
        assertRule("class Fixture { Fixture(int x) { " + body + " } }", NESTING, true);
    }

    @Test
    void compactConstructorsAreChecked() {
        String body = THREE_LEVELS.formatted("if (x > 1) { x--; }");
        assertRule("record Fixture(int x) { Fixture { " + body + " } }", NESTING, true);
    }

    @Test
    void initializersAreChecked() {
        String body = THREE_LEVELS.formatted("if (x > 1) { x--; }");
        assertRule("class Fixture { int x; { " + body + " } }", NESTING, true);
    }

    @Test
    void nestedMethodsStartTheirOwnNestingCount() {
        String local = "class Local { void run(int x) { if (x > 1) { x--; } } }";
        assertRule(method(THREE_LEVELS.formatted(local)), NESTING, false);
    }

    @Test
    void lambdasStartTheirOwnNestingCount() {
        String lambda = "Runnable action = () -> { if (true) { System.out.println(1); } };";
        assertRule(method(THREE_LEVELS.formatted(lambda)), NESTING, false);
    }

    @Test
    void deeplyNestedLambdaBodiesAreStillChecked() {
        String body = THREE_LEVELS.formatted("if (x > 1) { System.out.println(x); }");
        assertRule(method("Runnable action = () -> { " + body + " };"), NESTING, true);
    }

    private static String method(String body) {
        return "class Fixture { void run(int x) { " + body + " } }";
    }

    private static void assertRule(String source, String rule, boolean expected) {
        Report report = analyse(source);
        assertTrue(report.getProcessingErrors().isEmpty(), report.getProcessingErrors().toString());
        assertTrue(report.getConfigurationErrors().isEmpty(), report.getConfigurationErrors().toString());
        boolean found = report.getViolations().stream().anyMatch(item -> item.getRule().getName().equals(rule));
        assertEquals(expected, found, report.getViolations().toString());
    }

    private static Report analyse(String source) {
        PMDConfiguration configuration = new PMDConfiguration();
        configuration.addRuleSet("config/pmd.xml");
        configuration.setIgnoreIncrementalAnalysis(true);
        try (PmdAnalysis analysis = PmdAnalysis.create(configuration)) {
            analysis.files().addSourceFile(FileId.fromPathLikeString("Fixture.java"), source);
            Report report = analysis.performAnalysisAndCollectReport();
            assertEquals(0, analysis.getReporter().numErrors(), "PMD must not silently disable a broken rule");
            return report;
        }
    }
}
