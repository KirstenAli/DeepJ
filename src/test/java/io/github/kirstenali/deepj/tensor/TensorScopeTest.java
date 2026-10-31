package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

class TensorScopeTest {
    @Test
    void closedScopesReleasePrivateViewsEvenWhileTheirJavaReferencesRemainLive() {
        Tensor source = Tensor.ones(1, 4);
        Tensor alias;
        try (var ignored = TensorScope.open()) {
            alias = Tensor.aliasStorage(source);
            assertFalse(source.memory().canReleaseTemporary());
        }
        assertTrue(source.memory().canReleaseTemporary());
        assertSame(source.memory(), alias.memory());
        assertNull(TensorScope.current());
    }

    @Test
    void returningAnAliasFromTheOuterScopePreservesPublicOwnership() {
        Tensor alias;
        try (var scope = TensorScope.open()) {
            alias = scope.escape(Tensor.aliasStorage(Tensor.ones(1, 2)));
        }
        assertFalse(alias.memory().canReleaseTemporary());
    }

    @Test
    void aNestedReturnTransfersToItsCallerRatherThanBecomingPermanentlyPublic() {
        Tensor alias;
        try (var outer = TensorScope.open()) {
            try (var inner = TensorScope.open()) {
                alias = inner.escape(Tensor.aliasStorage(Tensor.ones(1, 2)));
            }
            assertSame(outer, TensorScope.current());
            assertFalse(alias.memory().canReleaseTemporary());
        }
        assertTrue(alias.memory().canReleaseTemporary());
    }

    @Test
    void anExternalViewOfPrivateStorageKeepsTheSharedAllocationAlive() {
        Tensor external;
        try (var ignored = TensorScope.open()) {
            Tensor internal = Tensor.aliasStorage(Tensor.ones(1, 2));
            external = TensorScope.external(() -> Tensor.aliasStorage(internal));
        }
        assertFalse(external.memory().canReleaseTemporary());
    }

    @Test
    void callbacksRestoreTheBoundaryWhenTheyThrowAndWhenNoBoundaryExists() {
        assertNull(TensorScope.external(TensorScope::current));
        try (var outer = TensorScope.open()) {
            assertThrows(IllegalArgumentException.class, () -> TensorScope.external(() -> {
                assertNull(TensorScope.current());
                throw new IllegalArgumentException("callback failed");
            }));
            assertSame(outer, TensorScope.current());
        }
        assertNull(TensorScope.current());
    }

    @Test
    void escapeNeverChangesOwnershipOfAnExistingPublicAlias() {
        Tensor alias = Tensor.aliasStorage(Tensor.ones(1, 2));
        try (var scope = TensorScope.open()) {
            assertSame(alias, scope.escape(alias));
            assertNotNull(scope.escape(Tensor.ones(1, 1)));
        }
        assertFalse(alias.memory().canReleaseTemporary());
    }

    @Test
    void savedBackwardInputsAndExplicitRetentionsOutlivePrivateScopes() {
        Tensor saved, result, retained;
        try (var ignored = TensorScope.open()) {
            saved = Tensor.aliasStorage(Tensor.ones(1, 2)).requiresGrad(true);
            result = saved.sumTensor();
            retained = Tensor.aliasStorage(Tensor.ones(1, 2)).retainDeviceBuffer();
        }
        assertFalse(saved.memory().canReleaseTemporary());
        assertFalse(retained.memory().canReleaseTemporary());
        result.backward();
        retained.releaseDeviceBufferRetention();
        assertTrue(saved.memory().canReleaseTemporary());
        assertTrue(retained.memory().canReleaseTemporary());
    }

    @Test
    void scopesRequireReverseOrderAndCannotEscapeAfterClosing() {
        var outer = TensorScope.open();
        try (var inner = TensorScope.open()) {
            assertThrows(IllegalStateException.class, outer::close);
            assertThrows(IllegalStateException.class, () -> outer.escape(Tensor.ones(1, 1)));
        }
        outer.close();
        outer.close();
        assertThrows(IllegalStateException.class, () -> outer.escape(Tensor.ones(1, 1)));
    }

    @Test
    void anotherThreadCannotCloseTheScope() throws InterruptedException {
        try (var scope = TensorScope.open()) {
            var failure = new AtomicReference<Throwable>();
            Thread thread = new Thread(() -> {
                assertNull(TensorScope.current());
                try { scope.close(); }
                catch (Throwable error) { failure.set(error); }
            });
            thread.start();
            thread.join();
            assertInstanceOf(IllegalStateException.class, failure.get());
            assertSame(scope, TensorScope.current());
        }
    }
}
