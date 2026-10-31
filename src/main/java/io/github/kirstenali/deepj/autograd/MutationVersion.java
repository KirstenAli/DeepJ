package io.github.kirstenali.deepj.autograd;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;

/** Shared only when aliases exist; weak state registrations do not retain their backward graphs. */
final class MutationVersion {
    private long version;
    private final List<WeakReference<GradientState>> states = new ArrayList<>();

    MutationVersion(long initialVersion) { version = initialVersion; }

    long value() { return version; }

    WeakReference<GradientState> track(GradientState state) {
        prune();
        var reference = new WeakReference<>(state);
        states.add(reference);
        return reference;
    }

    boolean any(Predicate<GradientState> predicate) {
        prune();
        return states.stream().map(WeakReference::get).filter(Objects::nonNull).anyMatch(predicate);
    }

    int trackedStates() { return states.size(); }

    private void prune() { states.removeIf(reference -> reference.get() == null); }

    void beforeMutation() {
        if (GradMode.isEnabled() && any(GradientState::required)) {
            throw new IllegalStateException("Use noGrad for parameter updates; an alias requires gradients");
        }
        version++;
    }
}
