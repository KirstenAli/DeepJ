package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public final class ModelSerializer {

    private ModelSerializer() {}

    public static void save(List<Parameter> params, Path path) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }

        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(path))) {
            out.writeInt(params.size());

            for (Parameter p : params) {
                Tensor t = p.value;
                out.writeInt(t.rows);
                out.writeInt(t.cols);

                for (int r = 0; r < t.rows; r++) {
                    for (int c = 0; c < t.cols; c++) {
                        out.writeDouble(t.get(r, c));
                    }
                }
            }
        }
    }

    public static void load(List<Parameter> params, Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(Files.newInputStream(path))) {
            int count = in.readInt();

            if (count != params.size()) {
                throw new IOException("Parameter count mismatch");
            }

            for (int i = 0; i < count; i++) {
                Parameter p = params.get(i);
                Tensor t = p.value;

                int rows = in.readInt();
                int cols = in.readInt();

                if (rows != t.rows || cols != t.cols) {
                    throw new IOException("Shape mismatch at parameter " + i);
                }

                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        t.set(r, c, in.readDouble());
                    }
                }
            }
        }
    }
}