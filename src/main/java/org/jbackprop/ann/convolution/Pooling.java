package org.jbackprop.ann.convolution;

public class Pooling {

    public static double[][] pool(double[][] input, int poolSize, int stride, PoolingType poolingType) {

        ConvolutionOperation pool = (i, j)->{
            int startRow = i * stride;
            int startCol = j * stride;
            int endRow = startRow + poolSize;
            int endCol = startCol + poolSize;

            return switch (poolingType) {

                case MAX -> getMaxValue(input, startRow, startCol, endRow, endCol);

                case AVERAGE -> getAverageValue(input, startRow, startCol, endRow, endCol);

                case GLOBAL -> getGlobalValue(input);
            };
        };

        return SlidingWindow.convolve(input, poolSize, stride, pool);
    }


    public static double getMaxValue(double[][] input, int startRow, int startCol, int endRow, int endCol) {
        double maxVal = Double.MIN_VALUE;

        for (int row = startRow; row < endRow; row++) {
            for (int col = startCol; col < endCol; col++) {
                maxVal = Math.max(maxVal, input[row][col]);
            }
        }

        return maxVal;
    }

    public static double getAverageValue(double[][] input, int startRow, int startCol, int endRow, int endCol) {
        double sum = 0.0;

        for (int row = startRow; row < endRow; row++) {
            for (int col = startCol; col < endCol; col++) {
                sum += input[row][col];
            }
        }

        return sum / ((endRow - startRow) * (endCol - startCol));
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