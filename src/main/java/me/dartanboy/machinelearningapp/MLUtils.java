package me.dartanboy.machinelearningapp;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MLUtils {

    public static int getPredictionFromSingleRowOutput(INDArray output, int classCount) {
        int largestIndex = 0;
        double largestValue = 0;
        for (int i = 0; i < classCount; i++) {
            double value = output.getColumn(i).getDouble(0);
            if (value > largestValue) {
                largestValue = value;
                largestIndex = i;
            }
        }

        return largestIndex;
    }

}
