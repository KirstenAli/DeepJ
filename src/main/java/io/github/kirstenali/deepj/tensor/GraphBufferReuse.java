package io.github.kirstenali.deepj.tensor;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

/** Reuses equal-sized, unreachable scratch buffers whose command lifetimes do not overlap. */
final class GraphBufferReuse {
    record Plan(List<int[]> allocations, int[] commands) {}

    private GraphBufferReuse() {}

    static Plan prepare(GraphExecution graph) {
        Map<Integer, Lifetime> lifetimes = candidates(graph);
        scan(graph.cmdStream, graph.cmdPos, lifetimes, null);
        Map<Integer, Integer> aliases = reuse(lifetimes);
        if (aliases.isEmpty()) return new Plan(graph.pendingAllocs, graph.cmdStream);
        int[] commands = Arrays.copyOf(graph.cmdStream, graph.cmdPos);
        scan(commands, graph.cmdPos, null, aliases);
        List<int[]> allocations = graph.pendingAllocs.stream()
                .filter(allocation -> !aliases.containsKey(allocation[0])).toList();
        return new Plan(allocations, commands);
    }

    private static Map<Integer, Lifetime> candidates(GraphExecution graph) {
        Set<Integer> uploads = new HashSet<>();
        for (int[] upload : graph.pendingUploadIds) {
            uploads.add(upload[0]);
        }
        Map<Integer, Lifetime> result = new HashMap<>();
        for (int[] allocation : graph.pendingAllocs) {
            int id = allocation[0];
            var reference = graph.bufIdToMemory.get(id);
            if (!uploads.contains(id) && (graph.backwardBufferIds.contains(id)
                    || reference != null && reference.get() == null)) {
                long key = ((long) GpuAllocations.type(allocation).storageCode() << 32) | allocation[1];
                result.put(id, new Lifetime(id, key));
            }
        }
        return result;
    }

    private static void scan(int[] commands, int length, Map<Integer, Lifetime> lifetimes,
                              Map<Integer, Integer> aliases) {
        int position = 0;
        while (position < length) {
            GraphOpMetadata.OpMeta metadata = metadata(commands[position], length - position);
            for (int offset : metadata.bufferArgOffsets()) {
                int index = position + offset, id = commands[index];
                if (lifetimes != null) visit(lifetimes.get(id), position);
                else commands[index] = aliases.getOrDefault(id, id);
            }
            position += metadata.stride();
        }
    }

    private static GraphOpMetadata.OpMeta metadata(int opcode, int remaining) {
        var metadata = GraphOpMetadata.OP_METADATA;
        if (opcode < 0 || opcode >= metadata.length || metadata[opcode] == null
                || metadata[opcode].stride() > remaining) {
            throw new IllegalArgumentException("Invalid GPU command stream");
        }
        return metadata[opcode];
    }

    private static void visit(Lifetime lifetime, int position) {
        if (lifetime == null) return;
        lifetime.first = Math.min(lifetime.first, position);
        lifetime.last = position;
    }

    private static Map<Integer, Integer> reuse(Map<Integer, Lifetime> lifetimes) {
        var ordered = lifetimes.values().stream().filter(lifetime -> lifetime.last >= 0)
                .sorted(Comparator.comparingInt((Lifetime value) -> value.first).thenComparingInt(value -> value.id))
                .toList();
        var active = new PriorityQueue<>(Comparator.comparingInt((Lifetime value) -> value.last)
                .thenComparingInt(value -> value.id));
        Map<Long, ArrayDeque<Integer>> available = new HashMap<>();
        Map<Integer, Integer> aliases = new HashMap<>();
        for (Lifetime lifetime : ordered) {
            expire(active, available, lifetime.first);
            var free = available.get(lifetime.size);
            if (free != null && !free.isEmpty()) {
                lifetime.physical = free.removeFirst();
                aliases.put(lifetime.id, lifetime.physical);
            }
            active.add(lifetime);
        }
        return aliases;
    }

    private static void expire(PriorityQueue<Lifetime> active,
                               Map<Long, ArrayDeque<Integer>> available, int first) {
        while (!active.isEmpty() && active.peek().last < first) {
            Lifetime expired = active.remove();
            available.computeIfAbsent(expired.size, ignored -> new ArrayDeque<>()).addLast(expired.physical);
        }
    }

    private static final class Lifetime {
        private final int id;
        private final long size;
        private int first = Integer.MAX_VALUE, last = -1, physical;

        private Lifetime(int id, long size) {
            this.id = id;
            this.size = size;
            physical = id;
        }
    }
}
