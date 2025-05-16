package org.DeepJ.convolution;

public class SlidingWindow {

    public static double[][] convolve(double[][] matrix, int filterSize, int stride,
                                Convolution convolution) {
        int numRows = matrix.length;
        int numCols = matrix[0].length;

        int outputRows = (numRows - filterSize) / stride + 1;
        int outputCols = (numCols - filterSize) / stride + 1;
        double[][] result = new double[outputRows][outputCols];

        for (int i = 0; i < outputRows; i++) {
            for (int j = 0; j < outputCols; j++) {
                result[i][j] = convolution.perform(i,j);
            }
        }

        return result;
    }
}

