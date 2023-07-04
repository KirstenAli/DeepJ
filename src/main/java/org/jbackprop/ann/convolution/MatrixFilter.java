package org.jbackprop.ann.convolution;

public class MatrixFilter {

    public static double[][] applyFilter(double[][] matrix, double[][] filter, int stride) {
        int filterSize = filter.length;

        ConvolutionOperation applyFilter = (i, j)-> {
            int sum = 0;
            for (int k = 0; k < filterSize; k++) {
                for (int l = 0; l < filterSize; l++) {
                    sum += matrix[i*stride+k][j*stride+l] * filter[k][l];
                }
            }

            return sum;
        };

        return SlidingWindow.convolve(matrix, filterSize, stride, applyFilter);
    }

    public static void main(String[] args){
        double[][] matrix1 = {
                {1, 2, 3, 1, 2, 3},
                {4, 5, 6, 1, 2, 3},
                {7, 8, 9, 1, 2, 3},
                {7, 8, 9, 1, 2, 3}
        };
        double[][] filter1 = {
                {1, 0},
                {0, 1},

        };
        int stride1 = 2;

        double[][] filteredMatrix1 = applyFilter(matrix1, filter1, stride1);

// Print the resulting filtered matrix
        for (int i = 0; i < filteredMatrix1.length; i++) {
            for (int j = 0; j < filteredMatrix1[0].length; j++) {
                System.out.print(filteredMatrix1[i][j] + " ");
            }
            System.out.println();
        }

    }

}
