# Tool-calling agents

`io.github.kirstenali.deepj.agent.Agent` takes a user request, advertises registered tools to the
model, executes structured calls and sends the results back so the model can answer. The caller
sets the total tool call budget per request. A reply can request multiple calls, which run in order.

`Qwen3AgentModel` uses DeepJ's Qwen3 chat model, tokenizer and generation session. Other model
adapters can implement `AgentModel` with the same Hermes format. Register custom tools by
implementing `Tool` with a name, description, JSON Schema for the arguments and a JSON result.
The tool implementation validates its arguments before acting.

## Terminal tool

The application supplies a working directory, an executable allowlist, a command policy and a
timeout. The model sends an `argv` array with the command name followed by individual arguments.
The policy receives the resolved executable path and each argument before any process starts.
The process receives EOF on stdin and an empty environment. Its combined stdout and stderr
are returned with the exit code, with capture limited to 8 KiB.

Approved executables run with the JVM user's permissions. The working directory sets where
commands start. Enforce any filesystem or network restrictions at the OS level, and authorize
specific arguments as well as executables.

```java
import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.agent.Qwen3AgentModel;
import io.github.kirstenali.deepj.agent.TerminalTool;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatModel;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;

// Run with a local Qwen3 snapshot on a system with /bin/pwd.
var chat = Qwen3ChatModel.load(Path.of("qwen3-1.7b"));
var terminal = new TerminalTool(Path.of("."),
        Map.of("pwd", Path.of("/bin/pwd")),
        argv -> argv.equals(List.of("/bin/pwd")),
        Duration.ofSeconds(5));
var agent = new Agent(new Qwen3AgentModel(chat), List.of(terminal), 3);
System.out.println(agent.run("Use the terminal to tell me the current directory."));
```
