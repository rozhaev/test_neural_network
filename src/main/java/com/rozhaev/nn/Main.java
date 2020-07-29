package com.rozhaev.nn;

public class Main {

    public static void main(String[] args){
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        try {
            int[][] inputs = {
                    {0,0,0},
                    {0,0,1},
                    {0,1,0},
                    {0,1,1},
                    {1,0,0},
                    {1,0,1},
                    {1,1,0},
                    {1,1,1},
            };

            int[][] targets = {
                    {0},
                    {1},
                    {1},
                    {0},
                    {1},
                    {0},
                    {0},
                    {1}
            };
            neuralNetwork.run(inputs, targets);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
