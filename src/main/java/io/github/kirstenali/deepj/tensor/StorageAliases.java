package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradientState;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;

/** Views keep their allocation alive without the allocation retaining abandoned backward graphs. */
final class StorageAliases {
    private final WeakReference<GradientState> root;
    private final List<WeakReference<GradientState>> views = new ArrayList<>();

    StorageAliases(GradientState state) { root = new WeakReference<>(state); }

    WeakReference<GradientState> add(GradientState state) {
        prune();
        var reference = new WeakReference<>(state);
        views.add(reference);
        return reference;
    }

    boolean liveViews() {
        prune();
        return !views.isEmpty();
    }

    int trackedViews() { return views.size(); }

    private void prune() { views.removeIf(reference -> reference.get() == null); }

    boolean savedForBackward() {
        if (saved(root.get())) return true;
        return views.stream().anyMatch(reference -> saved(reference.get()));
    }

    private static boolean saved(GradientState state) { return state != null && state.savedForBackward(); }
}
