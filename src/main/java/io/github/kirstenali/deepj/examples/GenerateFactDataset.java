package io.github.kirstenali.deepj.examples;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/** Generates deterministic question-and-answer records whose answers can be checked exactly. */
public final class GenerateFactDataset {

    private static final int MAXIMUM_LIMIT = 1_000;
    private static final String[] PLANETS = {
            "Mercury", "Venus", "Earth", "Mars",
            "Jupiter", "Saturn", "Uranus", "Neptune"
    };
    private static final Element[] ELEMENTS = {
            new Element("Hydrogen", "H"), new Element("Helium", "He"),
            new Element("Lithium", "Li"), new Element("Beryllium", "Be"),
            new Element("Boron", "B"), new Element("Carbon", "C"),
            new Element("Nitrogen", "N"), new Element("Oxygen", "O"),
            new Element("Fluorine", "F"), new Element("Neon", "Ne"),
            new Element("Sodium", "Na"), new Element("Magnesium", "Mg"),
            new Element("Aluminium", "Al"), new Element("Silicon", "Si"),
            new Element("Phosphorus", "P"), new Element("Sulfur", "S"),
            new Element("Chlorine", "Cl"), new Element("Argon", "Ar"),
            new Element("Potassium", "K"), new Element("Calcium", "Ca")
    };
    private static final Fact[] STABLE_FACTS = {
            new Fact("How many bits are in one byte?", "There are 8 bits in one byte."),
            new Fact("How many bytes are in one kibibyte?", "There are 1,024 bytes in one kibibyte."),
            new Fact("How many seconds are in one minute?", "There are 60 seconds in one minute."),
            new Fact("How many minutes are in one hour?", "There are 60 minutes in one hour."),
            new Fact("How many hours are in one day?", "There are 24 hours in one day."),
            new Fact("How many days are in one week?", "There are 7 days in one week."),
            new Fact("How many months are in one year?", "There are 12 months in one year."),
            new Fact("How many centimetres are in one metre?", "There are 100 centimetres in one metre."),
            new Fact("How many millimetres are in one metre?", "There are 1,000 millimetres in one metre."),
            new Fact("How many metres are in one kilometre?", "There are 1,000 metres in one kilometre."),
            new Fact("How many grams are in one kilogram?", "There are 1,000 grams in one kilogram.")
    };

    private GenerateFactDataset() {}

    public static void main(String[] args) throws IOException {
        Path output = Path.of(System.getProperty(
                "deepj.factsOutput", "checkpoints/knowledge-deepseek/facts.txt"));
        int maximum = Integer.getInteger("deepj.factMaximum", 100);
        GenerationResult result = generate(output, maximum);
        System.out.printf("Generated %,d checked fact records in %s%n", result.total(), output);
    }

    public static GenerationResult generate(Path output, int maximum) throws IOException {
        Objects.requireNonNull(output, "output");
        validateMaximum(maximum);
        Path parent = output.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
        try (BufferedWriter writer = Files.newBufferedWriter(output)) {
            long arithmetic = writeArithmetic(writer, maximum);
            long reference = writeReferenceFacts(writer);
            return new GenerationResult(arithmetic, reference);
        }
    }

    private static void validateMaximum(int maximum) {
        if (maximum < 1 || maximum > MAXIMUM_LIMIT) {
            throw new IllegalArgumentException("maximum must be between 1 and " + MAXIMUM_LIMIT);
        }
    }

    private static long writeArithmetic(BufferedWriter writer, int maximum) throws IOException {
        return writeAdditions(writer, maximum)
                + writeSubtractions(writer, maximum)
                + writeMultiplications(writer, maximum)
                + writeDivisions(writer, maximum);
    }

    private static long writeAdditions(BufferedWriter writer, int maximum) throws IOException {
        long count = 0;
        for (int left = 0; left <= maximum; left++) {
            for (int right = 0; right <= maximum; right++) {
                writeFact(writer, "What is %d plus %d?".formatted(left, right), left + right + ".");
                count++;
            }
        }
        return count;
    }

    private static long writeSubtractions(BufferedWriter writer, int maximum) throws IOException {
        long count = 0;
        for (int left = 0; left <= maximum; left++) {
            for (int right = 0; right <= left; right++) {
                writeFact(writer, "What is %d minus %d?".formatted(left, right), left - right + ".");
                count++;
            }
        }
        return count;
    }

    private static long writeMultiplications(BufferedWriter writer, int maximum)
            throws IOException {
        long count = 0;
        for (int left = 0; left <= maximum; left++) {
            for (int right = 0; right <= maximum; right++) {
                writeFact(writer, "What is %d multiplied by %d?".formatted(left, right),
                        left * right + ".");
                count++;
            }
        }
        return count;
    }

    private static long writeDivisions(BufferedWriter writer, int maximum) throws IOException {
        long count = 0;
        for (int divisor = 1; divisor <= maximum; divisor++) {
            for (int quotient = 0; quotient <= maximum; quotient++) {
                int dividend = divisor * quotient;
                writeFact(writer, "What is %d divided by %d?".formatted(dividend, divisor),
                        quotient + ".");
                count++;
            }
        }
        return count;
    }

    private static long writeReferenceFacts(BufferedWriter writer) throws IOException {
        long count = writePlanetFacts(writer) + writeElementFacts(writer);
        for (Fact fact : STABLE_FACTS) {
            writeFact(writer, fact.question(), fact.answer());
            count++;
        }
        return count;
    }

    private static long writePlanetFacts(BufferedWriter writer) throws IOException {
        for (int index = 0; index < PLANETS.length; index++) {
            String question = "Which planet is number %d from the Sun?".formatted(index + 1);
            writeFact(writer, question, PLANETS[index] + ".");
        }
        return PLANETS.length;
    }

    private static long writeElementFacts(BufferedWriter writer) throws IOException {
        long count = 0;
        for (int index = 0; index < ELEMENTS.length; index++) {
            Element element = ELEMENTS[index];
            writeFact(writer, "What is the chemical symbol for " + element.name() + "?",
                    element.symbol() + ".");
            writeFact(writer, "What is the atomic number of " + element.name() + "?",
                    index + 1 + ".");
            count += 2;
        }
        return count;
    }

    private static void writeFact(BufferedWriter writer, String question, String answer)
            throws IOException {
        writer.write("Instruction:\n");
        writer.write(question);
        writer.write("\nResponse:\n");
        writer.write(answer);
        writer.write("\n<|endoftext|>\n\n");
    }

    public record GenerationResult(long arithmetic, long reference) {
        public long total() {
            return arithmetic + reference;
        }
    }

    private record Element(String name, String symbol) {}

    private record Fact(String question, String answer) {}
}
