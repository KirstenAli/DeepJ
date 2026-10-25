package io.github.kirstenali.deepj.cli;

enum CliTheme {
    AURORA(81, new int[]{87, 81, 75, 99, 135, 141}),
    EMBER(215, new int[]{229, 222, 215, 209, 203, 204}),
    MONO(-1, new int[]{-1, -1, -1, -1, -1, -1});

    final int accent;
    final int[] gradient;

    CliTheme(int accent, int[] gradient) {
        this.accent = accent;
        this.gradient = gradient;
    }
}
