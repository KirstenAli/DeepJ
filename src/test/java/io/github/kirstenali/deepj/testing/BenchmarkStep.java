package io.github.kirstenali.deepj.testing;

import jdk.jfr.Event;
import jdk.jfr.Name;

@Name("deepj.BenchmarkStep")
final class BenchmarkStep extends Event {
    String phase;
    int round;

    static BenchmarkStep start(String phase, int round) {
        BenchmarkStep event = new BenchmarkStep();
        event.phase = phase;
        event.round = round;
        event.begin();
        return event;
    }

    void finish() {
        end();
        commit();
    }
}
