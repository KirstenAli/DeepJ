package io.github.kirstenali.deepj.examples;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FormatAlpacaTextTest {

    private static final String HEADER = "Below is an instruction that describes a task. "
            + "Write a response that appropriately completes the request.";
    private static final String CONTEXT_HEADER = "Below is an instruction that describes a task, "
            + "paired with an input that provides further context. Write a response that "
            + "appropriately completes the request.";

    @TempDir
    Path tempDir;

    @Test
    void formatsRecordsWithAndWithoutContext() throws Exception {
        String source = HEADER + "\n\nName a color.\n\nBlue.\n\n"
                + CONTEXT_HEADER + "\n\nCorrect the text.\n\nHelo\n\nHello.\n";
        Path input = write("input.txt", source);
        Path output = tempDir.resolve("output.txt");

        assertEquals(new FormatAlpacaText.FormatResult(2, 0),
                FormatAlpacaText.format(input, output));
        assertEquals(expectedRecords(), Files.readString(output));
    }

    @Test
    void preservesParagraphsInsideAResponse() throws Exception {
        Path input = write("input.txt", HEADER + "\n\nTell a story.\n\nFirst.\n\nSecond.\n");
        Path output = tempDir.resolve("output.txt");

        FormatAlpacaText.format(input, output);

        assertEquals("Instruction:\nTell a story.\nResponse:\nFirst.\n\nSecond.\n"
                + "<|endoftext|>\n\n", Files.readString(output));
    }

    @Test
    void skipsIncompleteRecords() throws Exception {
        Path input = write("input.txt", HEADER + "\n\nMissing response.\n");
        Path output = tempDir.resolve("output.txt");

        assertEquals(new FormatAlpacaText.FormatResult(0, 1),
                FormatAlpacaText.format(input, output));
        assertEquals("", Files.readString(output));
    }

    private Path write(String name, String value) throws IOException {
        return Files.writeString(tempDir.resolve(name), value);
    }

    private static String expectedRecords() {
        return "Instruction:\nName a color.\nResponse:\nBlue.\n<|endoftext|>\n\n"
                + "Instruction:\nCorrect the text.\nInput:\nHelo\nResponse:\nHello.\n"
                + "<|endoftext|>\n\n";
    }
}
