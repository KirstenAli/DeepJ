package io.github.kirstenali.deepj.agent;

import java.util.ArrayList;
import java.util.List;

/** Recent completed turns, with stable snapshots for generation and retries. */
public final class AgentHistory {
    private static final int MAX_TURNS = 12;
    private final ArrayList<Agent.Turn> turns = new ArrayList<>();

    public List<Agent.Turn> snapshot() { return List.copyOf(turns); }

    public void remember(Agent.Turn turn) {
        if (turn.reply() == null || turn.reply().isBlank()) return;
        if (turns.size() == MAX_TURNS) turns.remove(0);
        turns.add(turn);
    }

    public void clear() { turns.clear(); }
}
