package io.github.kirstenali.deepj.tensor.metal;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

final class NativeSourceFiles {
    private static final Set<String> EXTENSIONS = Set.of("h", "hpp", "c", "cc", "cpp", "m", "mm", "metal");

    private NativeSourceFiles() {}

    static List<Path> all() throws IOException {
        return under(List.of(Path.of("native"), Path.of("src/test/native")));
    }

    static List<Path> under(List<Path> roots) throws IOException {
        List<Path> sources = new ArrayList<>();
        for (Path root : roots) {
            try (Stream<Path> files = Files.walk(root)) {
                sources.addAll(files.filter(Files::isRegularFile).filter(NativeSourceFiles::isSource).toList());
            }
        }
        return sources.stream().sorted().toList();
    }

    private static boolean isSource(Path file) {
        String name = file.getFileName().toString();
        int separator = name.lastIndexOf('.');
        return separator > 0 && EXTENSIONS.contains(name.substring(separator + 1));
    }
}
