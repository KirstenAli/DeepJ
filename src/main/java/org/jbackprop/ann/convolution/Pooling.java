package org.jbackprop.ann.convolution;

public class Pooling {

    public static double[][] pool(double[][] input, int poolSize, int stride, PoolingType poolingType) {

        Convolution convolution = (i, j)->{
            int startRow = i * stride;
            int startCol = j * stride;
            int endRow = startRow + poolSize;
            int endCol = startCol + poolSize;

            return switch (poolingType) {

                case MAX -> getMaxValue(input, startRow, startCol, endRow, endCol);

                case AVERAGE -> getAverageValue(input, startRow, startCol, endRow, endCol);
            };
        };

        return SlidingWindow.convolve(input, poolSize, stride, convolution);
    }


    public static double getMaxValue(double[][] input, int startRow, int startCol, int endRow, int endCol) {
        final double[] maxVal = {Double.MIN_VALUE};

        WindowOperation windowOperation = (row, col)->
                maxVal[0] = Math.max(maxVal[0], input[row][col]);

        Window.apply(startRow, startCol, endRow, endCol, windowOperation);

        return maxVal[0];
    }

    public static double getAverageValue(double[][] input, int startRow, int startCol, int endRow, int endCol) {
        final double[] sum = {0};

        WindowOperation windowOperation = (row, col)->
                sum[0] += input[row][col];

        Window.apply(startRow, startCol, endRow, endCol, windowOperation);

        return sum[0] / ((endRow - startRow) * (endCol - startCol));
    }

    public static double getGlobalValue(double[][] input) {
        double sum = 0.0;
        int totalElements = input.length * input[0].length;

        for (double[] row : input) {
            for (double val : row) {
                sum += val;
            }
        }

        return sum / totalElements;
    }
}