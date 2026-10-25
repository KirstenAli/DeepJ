package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.persistence.ModelInspection;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

import java.io.InputStream;
import java.io.PrintWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.BooleanSupplier;

record CliServices(InputStream input, PrintWriter out, PrintWriter err, boolean interactive,
                   Map<String, String> environment, Models models, Consoles consoles,
                   Inspector inspector, BooleanSupplier metalAvailable) {
    static CliServices system() {
        return new CliServices(System.in, new PrintWriter(System.out, true), new PrintWriter(System.err, true),
                System.console() != null, System.getenv(), CliModel::load, CliConsole::system,
                path -> ModelSerializer.inspect(CliModel.modelFile(path)), MetalBackend::isAvailable);
    }

    @FunctionalInterface
    interface Models { CliModel load(Path path, BackendChoice backend) throws IOException; }

    @FunctionalInterface
    interface Consoles { CliConsole open(CliServices services) throws IOException; }

    @FunctionalInterface
    interface Inspector { ModelInspection inspect(Path path) throws IOException; }
}
