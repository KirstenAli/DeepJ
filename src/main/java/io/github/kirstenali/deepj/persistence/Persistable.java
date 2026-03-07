package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.Parameter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public interface Persistable {

    List<Parameter> parameters();

    default void save(Path path) throws IOException {
        ModelSerializer.save(parameters(), path);
    }

    default void load(Path path) throws IOException {
        ModelSerializer.load(parameters(), path);
    }
}