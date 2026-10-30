🚀 **Built My First RNN-Based Sentiment Analysis Model Using TensorFlow & Keras!**

I'm excited to share my progress in Deep Learning and Natural Language Processing (NLP).

I developed a binary sentiment classification model that predicts whether a given text expresses Positive or Negative sentiment.

🔹 **Project Workflow:**

1. Text Processing – Created a dataset containing positive and negative sentences.
2. Tokenization – Converted words into numerical sequences using Keras Tokenizer.
3. Padding – Standardized input sequence lengths using pad_sequences.
4. Embedding Layer – Converted word indices into 16-dimensional vectors.
5. SimpleRNN – Used 8 RNN units to learn sequential patterns.
6. Dense Layer – Applied Sigmoid activation for binary classification.
7. Model Compilation – Used Adam optimizer and Binary Crossentropy loss.
8. Model Training – Trained for 25 epochs with a batch size of 8.
9. Prediction – Tested new sentences and classified them as Positive or Negative.

📐 **Mathematical Understanding – Binary Crossentropy Loss**

For binary classification, the loss function is:

L = -[y log(p) + (1-y) log(1-p)]

Where:

* L = Loss
* y = Actual label (0 or 1)
* p = Predicted probability

The loss function measures the difference between actual labels and predicted probabilities.

For example:

* Actual label = 1, Prediction = 0.8
* Loss = -log(0.8) ≈ 0.2231

A lower loss indicates that the prediction is closer to the actual label.

📊 **Model Performance:**

* Training Accuracy: 100%
* Final Training Loss: 0.4721
* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Epochs: 25
* Batch Size: 8

💡 **Key Learnings:**

Through this project, I explored:

* Sequential text processing.
* Word embeddings.
* Recurrent Neural Networks.
* Binary classification.
* Mathematical understanding of loss functions.
* Model training and prediction.

Note: The model was trained on a small dataset of 30 sentences. The reported accuracy is training accuracy, not independent test accuracy. Further evaluation with a larger dataset is needed.

I'm looking forward to exploring more advanced concepts in Deep Learning, NLP, and Generative AI.

#DeepLearning #RNN #NLP #SentimentAnalysis #Python #TensorFlow #Keras #MachineLearning #ArtificialIntelligence
