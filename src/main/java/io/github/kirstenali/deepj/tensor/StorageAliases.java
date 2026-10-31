package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradientState;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;

/** Views keep their allocation alive without the allocation retaining abandoned backward graphs. */
final class StorageAliases {
    private final WeakReference<GradientState> root;
    private final List<Alias> views = new ArrayList<>();

    StorageAliases(GradientState state) { root = new WeakReference<>(state); }

    WeakReference<GradientState> add(GradientState state) {
        prune();
        var reference = new Alias(state, TensorScope.current());
        views.add(reference);
        return reference;
    }

    boolean liveViews() {
        prune();
        for (int index = 0; index < views.size(); index++) {
            if (views.get(index).protectsStorage()) return true;
        }
        return false;
    }

    int trackedViews() { return views.size(); }

    private void prune() { views.removeIf(reference -> reference.get() == null); }

    boolean savedForBackward() {
        if (saved(root.get())) return true;
        for (int index = 0; index < views.size(); index++) {
            if (saved(views.get(index).get())) return true;
        }
        return false;
    }

    private static boolean saved(GradientState state) { return state != null && state.savedForBackward(); }

    void escape(GradientState state, TensorScope source, TensorScope destination) {
        for (Alias alias : views) {
            if (alias.get() == state && alias.scope == source) alias.scope = destination;
        }
    }

    private static final class Alias extends WeakReference<GradientState> {
        private TensorScope scope;

        private Alias(GradientState state, TensorScope scope) {
            super(state);
            this.scope = scope;
        }

        private boolean protectsStorage() { return scope == null || scope.active(); }
    }
}
