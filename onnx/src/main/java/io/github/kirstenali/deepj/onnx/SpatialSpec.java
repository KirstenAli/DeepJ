package io.github.kirstenali.deepj.onnx;

import java.util.Arrays;

record SpatialSpec(int[] input, int[] kernel, int[] stride, int[] dilation, int[] before, int[] after, int[] output) {
    static SpatialSpec read(OnnxContext c, int[] input, int[] kernel) {
        int rank = input.length;
        int[] stride = c.integers("strides", ones(rank)), dilation = c.integers("dilations", ones(rank));
        int[] pads = c.integers("pads", new int[2 * rank]);
        if (kernel.length != rank
                || stride.length != rank
                || dilation.length != rank
                || pads.length != 2 * rank) {
            throw new IllegalArgumentException("Spatial attribute ranks differ");
        }
        String auto = c.string("auto_pad", "NOTSET");
        if (!auto.equals("NOTSET")) autoPad(input, kernel, stride, dilation, pads, auto);
        return create(input, kernel, stride, dilation, Arrays.copyOf(pads, rank),
                Arrays.copyOfRange(pads, rank, 2 * rank), c.integer("ceil_mode", 0) != 0);
    }

    static SpatialSpec create(int[] input, int[] kernel, int[] stride, int[] dilation, int[] before, int[] after,
            boolean ceil) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            requirePositive(kernel[i], stride[i], dilation[i]);
            long available = input[i] + (long) before[i] + after[i] - effective(kernel[i], dilation[i]);
            output[i] = Math.toIntExact((ceil ? -Math.floorDiv(-available, stride[i])
                                            : Math.floorDiv(available, stride[i]))
                                    + 1);
            if (ceil && (output[i] - 1L) * stride[i] >= input[i] + (long) before[i]) output[i]--;
            if (output[i] < 0)
                throw new IllegalArgumentException("Spatial kernel exceeds the padded input");
        }
        return new SpatialSpec(input.clone(), kernel.clone(), stride.clone(), dilation.clone(), before.clone(),
                after.clone(), output);
    }

    private static void autoPad(int[] input, int[] kernel, int[] stride, int[] dilation, int[] pads, String mode) {
        if (!java.util.Set.of("VALID", "SAME_UPPER", "SAME_LOWER").contains(mode))
            throw new IllegalArgumentException("Invalid auto_pad");
        for (int i = 0; i < input.length; i++) {
            int count = mode.equals("VALID") ? 0 : Math.max(0, (int)
                                            (((input[i] + (long) stride[i] - 1) / stride[i] - 1)
                                                            * stride[i]
                                                    + effective(kernel[i], dilation[i])
                                                    - input[i]));
            pads[i] = mode.equals("SAME_LOWER") ? (count + 1) / 2 : count / 2;
            pads[i + input.length] = count - pads[i];
        }
    }

    static int effective(int kernel, int dilation) {
        return Math.addExact(1, Math.multiplyExact(kernel - 1, dilation));
    }

    static int[] ones(int count) {
        int[] values = new int[count];
        Arrays.fill(values, 1);
        return values;
    }

    int positions() {
        return OnnxValue.elements(output);
    }

    int area() {
        return OnnxValue.elements(kernel);
    }

    int[] indices() {
        int[] indices = new int[Math.multiplyExact(positions(), area())];
        for (int position = 0; position < positions(); position++) {
            for (int tap = 0; tap < area(); tap++) {
                indices[position * area() + tap] = index(position, tap, false);
            }
        }
        return indices;
    }

    int index(int position, int tap, boolean padded) {
        int address = 0, multiplier = 1;
        for (int axis = input.length - 1; axis >= 0; axis--) {
            int coordinate = position % output[axis] * stride[axis]
                            - before[axis]
                            + tap % kernel[axis] * dilation[axis];
            position /= output[axis];
            tap /= kernel[axis];
            if (coordinate < (padded ? -before[axis] : 0)
                    || coordinate >= input[axis] + (padded ? after[axis] : 0)) return 0;
            address += coordinate * multiplier;
            multiplier *= input[axis];
        }
        return padded ? 1 : address + 1;
    }

    float[] counts(boolean includePadding) {
        float[] counts = new float[positions()];
        for (int i = 0; i < counts.length; i++) {
            for (int j = 0; j < area(); j++) {
                if (index(i, j, includePadding) != 0) counts[i]++;
            }
        }
        return counts;
    }

    private static void requirePositive(int kernel, int stride, int dilation) {
        if (kernel < 1 || stride < 1 || dilation < 1) {
            throw new IllegalArgumentException("Spatial sizes must be positive");
        }
    }
}
