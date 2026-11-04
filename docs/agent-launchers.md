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
find the current directory or list its files. When it requests a command, the dialog shows
the executable and working directory. Choose **Run once** or **Deny**.

## CLI

```bash
mvn package -Pcli
java -Xmx12g --enable-native-access=ALL-UNNAMED -cp target/deepj-cli.jar \
  io.github.kirstenali.deepj.cli.AgentCli ./qwen3-1.7b --workdir . --backend metal --dtype float32
```

Use `--backend cpu` to run on CPU. In an interactive terminal, answer `y` to approve a
requested command once. Enter defaults to denial. Use `/exit` to close the chat.

The interactive launchers expose `pwd` and `ls` without arguments on systems that provide
`/bin/pwd` and `/bin/ls`. Every call requires fresh approval. The [Java agent API](agents.md)
also lets applications register their own tools and define their own command policies.
