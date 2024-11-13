package me.dartanboy.machinelearningapp;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public abstract class MachineLearner {
    protected final int featureCount;
    protected final int classCount;
    protected final int batchSize;

    protected MultiLayerNetwork model;
    protected DataNormalization normalizer;

    private void createModel() {
        long seed = new Random().nextLong();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(featureCount).nOut(3)
                        .build())
                .layer(new DenseLayer.Builder().nIn(3).nOut(3)
                        .build())
                .layer( new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(classCount).build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
    }

    public MachineLearner(int featureCount, int classCount, int batchSize) {
        this.featureCount = featureCount;
        this.classCount = classCount;
        this.batchSize = batchSize;

        createModel();
    }

    public void train(RecordReader recordReader, FileSplit fileSplit) throws IOException, InterruptedException {
        // Initialize record reader
        recordReader.initialize(fileSplit);

        // Get data
        DataSetIterator iterator =
                new RecordReaderDataSetIterator(recordReader, batchSize, featureCount, classCount);
        DataSet trainingData = iterator.next();
        trainingData.shuffle();

        // Normalize data
        normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);
        normalizer.transform(trainingData);

        // Fit the model to the training data
        for(int i=0; i<1000; i++ ) {
            model.fit(trainingData);
        }
    }

    public void predictFromString(String input) {
        try {
            // Get the input
            List<List<String>> listList = new ArrayList<>();
            List<String> inputList = new ArrayList<>(Arrays.asList(input.split(",")));
            listList.add(inputList);
            ListStringSplit listStringSplit = new ListStringSplit(listList);
            ListStringRecordReader inputReader = new ListStringRecordReader();
            inputReader.initialize(listStringSplit);

            // Normalize the input
            DataSetIterator iterator =
                    new RecordReaderDataSetIterator(inputReader, 1);
            DataSet normalizedInputData = iterator.next();
            normalizer.transform(normalizedInputData);

            // Get the output
            INDArray output = model.output(normalizedInputData.getFeatures());

            printPrediction(output);
        }  catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    protected abstract void printPrediction(INDArray output);

    public void predictFromFile(String inputFileName) {
        try {
            // Get the input
            File file = new ClassPathResource(inputFileName).getFile();
            RecordReader inputReader = new CSVRecordReader(0, ',');
            inputReader.initialize(new FileSplit(file));

            // Normalize the input
            DataSetIterator iterator =
                    new RecordReaderDataSetIterator(inputReader, 1);
            DataSet normalizedInputData = iterator.next();
            normalizer.transform(normalizedInputData);

            // Get the output
            INDArray output = model.output(normalizedInputData.getFeatures());
            System.out.println("Weights for Prediction: " + output);
            int prediction = MLUtils.getPredictionFromSingleRowOutput(output, classCount);
            System.out.println("Prediction: " + (prediction == 0 ? "Free" : "Premium"));
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}
