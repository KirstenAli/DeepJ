package io.github.kirstenali.deepj.tensor;

import java.lang.ref.WeakReference;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

abstract class GraphOpMetadata {

    // Keep these values synchronized with deepj_metal_jni.mm.
    public static final int OP_ADD             = 1;
    public static final int OP_SUBTRACT        = 2;
    public static final int OP_MULTIPLY        = 3;
    public static final int OP_DIVIDE          = 4;
    public static final int OP_MATMUL          = 5;
    public static final int OP_MULTIPLY_SCALAR = 6;
    public static final int OP_SQRT            = 7;
    public static final int OP_NEG             = 8;
    public static final int OP_EXP             = 9;
    public static final int OP_LOG             = 10;
    public static final int OP_TANH            = 11;
    public static final int OP_SIGMOID         = 12;
    public static final int OP_RELU            = 13;
    public static final int OP_RELU_BACKWARD   = 14;
    public static final int OP_GELU            = 15;
    public static final int OP_GELU_BACKWARD   = 16;
    public static final int OP_SOFTMAX_ROWS    = 17;
    public static final int OP_SOFTMAX_BACKWARD= 18;
    public static final int OP_LAYERNORM_BACKWARD = 19;
    public static final int OP_ADAMW_UPDATE    = 20;
    public static final int OP_ADD_SCALAR      = 21;
    public static final int OP_DIVIDE_SCALAR   = 22;
    public static final int OP_TRANSPOSE       = 23;
    public static final int OP_ADD_ROW_VECTOR  = 24;
    public static final int OP_ADD_BROADCAST_COLS = 25;
    public static final int OP_SUBTRACT_BROADCAST_COLS = 26;
    public static final int OP_DIVIDE_BROADCAST_COLS = 27;
    public static final int OP_MULTIPLY_BROADCAST_ROWS = 28;
    public static final int OP_SUM_ROWS        = 29;
    public static final int OP_MEAN_ALONG_ROWS = 30;
    public static final int OP_VARIANCE_ALONG_ROWS = 31;
    public static final int OP_MULTIPLY_BROADCAST_COLS = 32;
    public static final int OP_SUM_ALONG_ROWS = 33;
    public static final int OP_MAX_ALONG_ROWS = 34;
    public static final int OP_CLAMP = 35;
    public static final int OP_POW = 36;
    public static final int OP_SCATTER_ADD_ROWS = 37;
    public static final int OP_SUM_ABS = 38;
    public static final int OP_CROSS_ENTROPY_LOSS = 39;
    public static final int OP_CROSS_ENTROPY_GRADIENT = 40;
    public static final int OP_SUM_SCALAR = 41;
    public static final int OP_SCATTER_ADD_ROWS_ATOMIC = 42;
    public static final int OP_SUM_SQUARES = 43;
    public static final int OP_BATCHED_MATMUL = 44;
    public static final int OP_SPLIT_HEADS = 45;
    public static final int OP_MERGE_HEADS = 46;
    public static final int OP_CAUSAL_MASK = 47;
    public static final int OP_ROTARY = 48;
    public static final int OP_GATHER_ROWS = 49;
    public static final int OP_CAUSAL_SOFTMAX = 50;
    public static final int OP_CROSS_ENTROPY_FUSED = 51;
    public static final int OP_SUM_SQUARES_SCALAR = 52;
    public static final int OP_RMS_NORM = 53;
    public static final int OP_RMS_NORM_BACKWARD = 54;
    public static final int OP_SWIGLU = 55;
    public static final int OP_SWIGLU_BACKWARD = 56;

    record OpMeta(int stride, int[] bufferArgOffsets) {}

    static final OpMeta[] OP_METADATA = buildOpMetadata();

    static OpMeta[] buildOpMetadata() {
        OpMeta[] meta = new OpMeta[OP_SWIGLU_BACKWARD + 1];
        registerUnaryMeta(meta);
        registerBinaryMeta(meta);
        registerReductionMeta(meta);
        registerLossMeta(meta);
        registerAttentionMeta(meta);
        registerBroadcastMeta(meta);
        registerComplexMeta(meta);
        return meta;
    }

    static void registerUnaryMeta(OpMeta[] meta) {
        registerMeta(meta, OP_SQRT, 4, 1, 2);
        registerMeta(meta, OP_NEG, 4, 1, 2);
        registerMeta(meta, OP_EXP, 4, 1, 2);
        registerMeta(meta, OP_LOG, 4, 1, 2);
        registerMeta(meta, OP_TANH, 4, 1, 2);
        registerMeta(meta, OP_SIGMOID, 4, 1, 2);
        registerMeta(meta, OP_RELU, 4, 1, 2);
        registerMeta(meta, OP_GELU, 4, 1, 2);
    }

    static void registerBinaryMeta(OpMeta[] meta) {
        registerMeta(meta, OP_ADD, 5, 1, 2, 3);
        registerMeta(meta, OP_SUBTRACT, 5, 1, 2, 3);
        registerMeta(meta, OP_MULTIPLY, 5, 1, 2, 3);
        registerMeta(meta, OP_DIVIDE, 5, 1, 2, 3);
        registerMeta(meta, OP_RELU_BACKWARD, 5, 1, 2, 3);
        registerMeta(meta, OP_GELU_BACKWARD, 5, 1, 2, 3);
    }

    static void registerReductionMeta(OpMeta[] meta) {
        registerScalarAndShapeMeta(meta);
        registerRowReductionMeta(meta);
        registerAggregateMeta(meta);
    }

    static void registerScalarAndShapeMeta(OpMeta[] meta) {
        registerMeta(meta, OP_MULTIPLY_SCALAR, 5, 1, 2);
        registerMeta(meta, OP_ADD_SCALAR, 5, 1, 2);
        registerMeta(meta, OP_DIVIDE_SCALAR, 5, 1, 2);
        registerMeta(meta, OP_TRANSPOSE, 5, 1, 2);
        registerMeta(meta, OP_CLAMP, 6, 1, 2);
        registerMeta(meta, OP_POW, 5, 1, 2);
    }

    static void registerRowReductionMeta(OpMeta[] meta) {
        registerMeta(meta, OP_SUM_ROWS, 5, 1, 2);
        registerMeta(meta, OP_MEAN_ALONG_ROWS, 5, 1, 2);
        registerMeta(meta, OP_VARIANCE_ALONG_ROWS, 5, 1, 2);
        registerMeta(meta, OP_SUM_ALONG_ROWS, 5, 1, 2);
        registerMeta(meta, OP_MAX_ALONG_ROWS, 5, 1, 2);
        registerMeta(meta, OP_SOFTMAX_ROWS, 5, 1, 2);
    }

    static void registerAggregateMeta(OpMeta[] meta) {
        registerMeta(meta, OP_SCATTER_ADD_ROWS, 7, 1, 2, 3);
        registerMeta(meta, OP_SUM_ABS, 5, 1, 2);
        registerMeta(meta, OP_SUM_SQUARES, 5, 1, 2);
        registerMeta(meta, OP_SUM_SCALAR, 5, 1, 2);
        registerMeta(meta, OP_SUM_SQUARES_SCALAR, 4, 1, 2);
    }

    static void registerLossMeta(OpMeta[] meta) {
        registerMeta(meta, OP_CROSS_ENTROPY_LOSS, 6, 1, 2, 3);
        registerMeta(meta, OP_CROSS_ENTROPY_GRADIENT, 6, 1, 2, 3);
        registerMeta(meta, OP_CROSS_ENTROPY_FUSED, 7, 1, 2, 3, 4);
        registerMeta(meta, OP_SCATTER_ADD_ROWS_ATOMIC, 7, 1, 2, 3);
    }

    static void registerAttentionMeta(OpMeta[] meta) {
        registerMeta(meta, OP_BATCHED_MATMUL, 11, 1, 2, 3);
        registerMeta(meta, OP_SPLIT_HEADS, 7, 1, 2);
        registerMeta(meta, OP_MERGE_HEADS, 7, 1, 2);
        registerMeta(meta, OP_CAUSAL_MASK, 5, 1, 2);
        registerMeta(meta, OP_ROTARY, 9, 1, 2, 3, 4);
        registerMeta(meta, OP_GATHER_ROWS, 7, 1, 2, 3);
        registerMeta(meta, OP_CAUSAL_SOFTMAX, 7, 1, 2);
    }

    static void registerBroadcastMeta(OpMeta[] meta) {
        registerMeta(meta, OP_SOFTMAX_BACKWARD, 6, 1, 2, 3);
        registerMeta(meta, OP_ADD_ROW_VECTOR, 6, 1, 2, 3);
        registerMeta(meta, OP_ADD_BROADCAST_COLS, 6, 1, 2, 3);
        registerMeta(meta, OP_SUBTRACT_BROADCAST_COLS, 6, 1, 2, 3);
        registerMeta(meta, OP_DIVIDE_BROADCAST_COLS, 6, 1, 2, 3);
        registerMeta(meta, OP_MULTIPLY_BROADCAST_ROWS, 6, 1, 2, 3);
        registerMeta(meta, OP_MULTIPLY_BROADCAST_COLS, 6, 1, 2, 3);
    }

    static void registerComplexMeta(OpMeta[] meta) {
        registerMeta(meta, OP_MATMUL, 7, 1, 2, 3);
        registerMeta(meta, OP_LAYERNORM_BACKWARD, 7, 1, 2, 3, 4);
        registerMeta(meta, OP_ADAMW_UPDATE, 13, 1, 2, 3, 4);
        registerMeta(meta, OP_RMS_NORM, 9, 1, 2, 3, 4, 5);
        registerMeta(meta, OP_RMS_NORM_BACKWARD, 8, 1, 2, 3, 4, 5);
        registerMeta(meta, OP_SWIGLU, 5, 1, 2, 3);
        registerMeta(meta, OP_SWIGLU_BACKWARD, 7, 1, 2, 3, 4, 5);
    }

    static void registerMeta(OpMeta[] meta, int op, int stride, int... bufferArgOffsets) {
        meta[op] = new OpMeta(stride, bufferArgOffsets);
    }


}
