# Manual-backprop baseline

`manual-baseline.bin` was captured on the CPU from unmodified master commit
`374f8464f24bd95ff61ffbe987b58198106b1d75`, before implementing autograd.
It contains 4,551 big-endian float32 values, with no model weights from Frisky or Nova training.

`MigrationBaselineTest` defines the exact model configurations, inputs and seeds.
For each of Origin, Orbit, Prism, Nova and the feed-forward model, the fixture stores:

1. The initial forward output.
2. Every parameter gradient for the fixed upstream gradient, in parameter order.
3. The loss and every updated parameter after each of three AdamW updates.

The transformer updates accumulate two different sequences per optimizer step.
The comparison uses the fixture, not a second copy of the old backward implementation.
Changes to gradient formulas, parameter ordering, averaging or optimizer sequencing must not
be accepted merely by regenerating this fixture.
