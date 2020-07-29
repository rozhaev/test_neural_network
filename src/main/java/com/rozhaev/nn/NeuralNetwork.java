package com.rozhaev.nn;

import java.util.Arrays;

public class NeuralNetwork {

    private int step = 0;
    private final int[] elements;
    private final double[][] activations;
    private double[][][] weights;
    private double error;

    public NeuralNetwork() {
        this.elements = new int[4];
        this.error = 0.5;
        this.activations = new double[4][];
        this.initState();
    }

    public void run(int[][] inputs, int[][] targets) throws Exception {

        float p = 0.5f;
        double[] errors = new double[inputs.length];
        do{
            for (int i = 0; i<inputs.length; i++){
                update(inputs[i]);
                errors[i] = backPropagate(targets[i], p);
                System.out.println("Step: "+step);
                step++;
            }
            error = Arrays.stream(errors).max().getAsDouble();
        } while (error >= 0.05);
    }

    private void initState(){
        elements[0] = 3;
        elements[1] = 4;
        elements[2] = 4;
        elements[3] = 1;

        this.activations[0] = new double[3];
        this.activations[1] = new double[4];
        this.activations[2] = new double[4];
        this.activations[3] = new double[1];

        this.weights = new double[4][][];
        this.weights[0] = new double[3][4];
        this.weights[1] = new double[4][4];
        this.weights[2] = new double[4][1];

        for (int l = 0; l < elements.length - 1; ++l) {
            for (int i = 0; i < elements[l]; ++i) {
                for (int j = 0; j < elements[l + 1]; ++j) {
                    this.weights[l][i][j] = rand(-1, 1);
                }
            }
        }
    }

    private double[] update(int[] inputs) throws Exception {
        if (inputs.length != this.elements[0]) throw new Exception("Input size is wrong.");
        int  l, i, j;

        for (i = 0; i < inputs.length; ++i)
            this.activations[0][i] = inputs[i];
        for (l = 0; l < this.elements.length - 1; ++l) {
            for (j = 0; j < this.elements[l + 1]; ++j) {
                double sum = 0;
                for (i = 0; i < this.elements[l]; ++i) {
                    sum += this.activations[l][i] * this.weights[l][i][j];
                }
                this.activations[l + 1][j] = sigmoid(sum);
            }
        }

        return this.activations[this.activations.length - 1];
    }

    private double backPropagate(int[] target, float p) throws Exception {
        if (target.length != this.elements[this.elements.length - 1]) throw new Exception("Input size is wrong.");

        double[][] act = this.activations;
        double[][][] w = this.weights;
        int l, i, j;

        double[][] delta = new double[4][];
        delta[0] = new double[3];
        delta[1] = new double[4];
        delta[2] = new double[4];
        delta[3] = new double[1];

        for (l = 0; l < this.elements.length; ++l) {
            for (i = 0; i < this.elements[l]; ++i) {
                delta[l][i] = 0;
            }
        }

        // hidden -> output
        int o = this.elements.length - 1;
        for (i = 0; i < this.elements[o]; ++i) {
            delta[o][i] = (target[i] - act[o][i]) * dsigm(act[o][i]);
        }

        // hidden or input -> hidden
        for (l = this.elements.length - 2; l > 0; --l) {
            for (i = 0; i < this.elements[l]; ++i) {
                for (j = 0; j < this.elements[l + 1]; ++j) {
                    delta[l][i] += delta[l + 1][j] * w[l][i][j];
                }
                delta[l][i] *= dsigm(act[l][i]);
            }
        }

        // update weights
        for (l = 0; l < this.elements.length - 1; ++l) {
            for (i = 0; i < this.elements[l]; ++i) {
                for (j = 0; j < this.elements[l + 1]; ++j) {
                    w[l][i][j] = w[l][i][j] + p * delta[l + 1][j] * act[l][i];
                }
            }
        }

        double error = 0;
        for (i = 0; i < target.length; ++i){
            error = Math.max(Math.abs(target[i] - act[o][i]), error);
        }

        return error;
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double dsigm(double y) {
        return y * (1 - y);
    }

    double rand(int a, int b) {
        return (b - a) * Math.random() + a;
    }
}
