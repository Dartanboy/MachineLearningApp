package me.dartanboy.machinelearningapp;

import org.nd4j.linalg.api.ndarray.INDArray;

public class PremiumPluginPredictor extends MachineLearner {
    public PremiumPluginPredictor() {
        super(4, 2, 15);
    }

    @Override
    protected void printPrediction(INDArray output) {
        System.out.println("Prediction Weights: " + output);
        int prediction = MLUtils.getPredictionFromSingleRowOutput(output, classCount);
        System.out.println("This plugin is probably: " + (prediction == 0 ? "Free" : "Premium"));
    }
}
