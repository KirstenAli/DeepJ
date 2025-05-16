package org.DeepJ.convolution;

public class Window {

    public static void apply(int startRow, int startCol, int endRow, int endCol,
                      WindowOperation windowOperation){
        
        for (int row = startRow; row < endRow; row++) {
            for (int col = startCol; col < endCol; col++) {
                windowOperation.apply(row, col);
            }
        }
    }
}
