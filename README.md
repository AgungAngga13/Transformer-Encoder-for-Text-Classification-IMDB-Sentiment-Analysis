# Transformer Encoder for Text Classification: IMDB Sentiment Analysis
This project implements a Transformer Encoder architecture from scratch using TensorFlow/Keras to perform sentiment classification on the IMDB movie reviews dataset. The implementation follows the original "Attention is All You Need" paper concepts, adapted for a smaller-scale educational experiment.

## 🚀 Key Features

Custom Transformer Block: Implementation of Multi-Head Attention, Layer Normalization, and Feed-Forward networks.


Positional Embedding: Incorporating word order information into the parallelized Transformer architecture.


Comparative Study: An ablation study comparing a baseline model against variations in attention heads, feed-forward dimensions, and training data volume.

## 📊 Experimental Results
The project analyzes how model complexity affects performance on limited datasets. Key findings include:


Baseline Performance: Achieved a test accuracy of 83.19%.


Overfitting Analysis: Doubling the Feed-Forward dimension (FF_DIM) led to significant overfitting, with training accuracy reaching 96.25% while test accuracy dropped to 78.70%.


Data Sensitivity: Reducing the training samples by just 1,000 records notably decreased the model's ability to generalize.

## 🛠️ Tech Stack
Language: Python

Framework: TensorFlow / Keras

Data Processing: NumPy, Pandas

Visualization: Matplotlib
