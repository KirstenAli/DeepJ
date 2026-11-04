package io.github.kirstenali.deepj.agent;

import java.io.IOException;
import java.util.Map;

/** A named operation with a JSON object schema and a JSON-serializable result. */
public interface Tool {
    String name();

    String description();

    Map<String, Object> parameters();

    /** Validates the supplied arguments before performing the operation. */
    Map<String, Object> execute(Map<String, Object> arguments) throws IOException, InterruptedException;
}
