package io.github.kirstenali.deepj.distributed;

/** Loss is a global contribution-weighted mean; step counts optimizer updates, not local microbatches. */
public record DistributedStep(long step, long samples, float loss) {}
