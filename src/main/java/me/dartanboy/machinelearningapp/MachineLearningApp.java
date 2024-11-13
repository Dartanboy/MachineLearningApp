package me.dartanboy.machinelearningapp;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.nd4j.common.io.ClassPathResource;
import java.io.IOException;
import java.util.Scanner;

public class MachineLearningApp {

    public static void main(String[] args) throws IOException, InterruptedException {
        MachineLearner machineLearner = new PremiumPluginPredictor();

        RecordReader recordReader = new CSVRecordReader(1, ',');
        FileSplit fileSplit = new FileSplit(new ClassPathResource("plugins.txt").getFile());

        machineLearner.train(recordReader, fileSplit);

        System.out.println("Please input your plugin's information in the following format:");
        System.out.println("days,downloads,rating,ratingCount");
        System.out.println("For example, a plugin that has been out for 3 days, " +
                "got 10 downloads, is rated 4.5 stars, and has 10 ratings, would look like this:");
        System.out.println("3,10,4.5,10");

        Scanner scanner = new Scanner(System.in);
        String line = scanner.nextLine();
        machineLearner.predictFromString(line);
        scanner.close();
    }
}
