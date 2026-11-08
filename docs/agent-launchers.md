# DeepJ Agent launchers

Both launchers use DeepJ's local models. Qwen3 is the first model with a tool-calling adapter.
Other supported checkpoints open in regular chat.

## Desktop

```bash
JAVA_TOOL_OPTIONS="-Xmx3g" mvn compile javafx:run
```

Choose the Qwen3 snapshot's `config.json` in the model picker. After the model loads, open
**Chat options** beside the message field, turn on **Agent mode**, and change the **Working directory**
if needed. Ask the model to run a command. The approval dialog shows the resolved executable,
each argument and the working directory. Choose **Run once** or **Deny**. To revise the command,
choose **Edit command**, update the executable or arguments, then choose **Review command**
to preview what will run. DeepJ includes the edited command with the output it sends back to the model.

The desktop **Thinking** switch sits beside Agent mode in the Chat options menu. It is off by default
and works independently of Agent mode. Thinking uses the generation token budget; allow enough
maximum tokens for reasoning and an answer. Both modes show reasoning in the expandable **Thought process**
above the answer. In Agent mode, the panel includes reasoning before and after tool calls.
Terminal commands still require the same approval.

The regular desktop application includes Agent mode for Qwen3. The older
`AgentDesktopApp` entry point also opens the same interface. Agent mode and
Thinking both start off. Desktop and CLI responses allow up to 2,048 generated
tokens by default, shared with reasoning when Thinking is enabled.

Qwen chat follows the checkpoint precision by default: BF16 weights and activations
for the official Qwen3-1.7B files. FP16 and FP32 remain selectable through the Qwen dtype setting.

## CLI

```bash
mvn package -Pcli
java -Xmx3g --enable-native-access=ALL-UNNAMED -cp target/deepj-cli.jar \
  io.github.kirstenali.deepj.cli.AgentCli ./qwen3-1.7b --workdir . --backend metal --thinking
```

Add `--thinking` to enable Qwen3 reasoning with tool calls. Reasoning streams before
each command approval and the final answer. Use `/set thinking on` or
`/set thinking off` between turns; `/settings` shows the current value. Thinking
starts off when the flag is omitted and shares each model response's token budget.

Use `--backend cpu` to run on CPU. In an interactive terminal, answer `y` to approve a
requested command once. Enter defaults to denial. Use `/exit` to close the chat.

Conversation history starts off. Use `/set history on` to include recent turns and their tool
results in follow-up prompts. Use `/set history off` to stop including earlier turns, or
`/settings` to check the current value.

On macOS and Linux, Agent mode accepts executables on the launcher's `PATH` and explicit paths
such as `./mvnw`. For pipes or redirects, the agent can request an explicit shell command.
The approval prompt includes the entire script as an argument.

Each approved command runs with your account's permissions and the launcher's environment.
Commands have a two-minute timeout, closed stdin, and an 8 KiB limit on captured stdout and
stderr combined. The [Java agent API](agents.md) lets applications define their own tool policies.
