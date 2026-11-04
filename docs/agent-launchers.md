# DeepJ Agent launchers

Both launchers use DeepJ's local models. Qwen3 is the first model with a tool-calling adapter.
Other supported checkpoints open in regular chat.

## Desktop

```bash
MAVEN_OPTS="-Xmx12g --enable-native-access=ALL-UNNAMED" \
  mvn compile exec:java -Dexec.mainClass=io.github.kirstenali.deepj.chatui.AgentDesktopApp
```

Choose the Qwen3 snapshot's `config.json` in the model picker. After the model loads, select
**Agent mode** in the toolbar and set the **Working directory** if needed. Ask the model to
run a command. The approval dialog shows the resolved executable, each argument and the
working directory. Choose **Run once** or **Deny**.

## CLI

```bash
mvn package -Pcli
java -Xmx12g --enable-native-access=ALL-UNNAMED -cp target/deepj-cli.jar \
  io.github.kirstenali.deepj.cli.AgentCli ./qwen3-1.7b --workdir . --backend metal --dtype float32
```

Use `--backend cpu` to run on CPU. In an interactive terminal, answer `y` to approve a
requested command once. Enter defaults to denial. Use `/exit` to close the chat.

On macOS and Linux, Agent mode accepts executables on the launcher's `PATH` and explicit paths
such as `./mvnw`. For pipes or redirects, the agent can request an explicit shell command.
The approval prompt includes the entire script as an argument.

Each approved command runs with your account's permissions and the launcher's environment.
Commands have a two-minute timeout, closed stdin, and an 8 KiB limit on captured stdout and
stderr combined. The [Java agent API](agents.md) lets applications define their own tool policies.
