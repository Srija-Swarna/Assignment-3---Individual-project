# Assignment-3---Individual-project

## **Overview**
This project involves using BERT (Bidirectional Encoder Representations from Transformers) for sequence classification on a dataset related to AG News. The aim is to train a model that can classify news articles into one of four predefined categories. This README provides a detailed explanation of the dataset used, the steps involved in preprocessing the data, training the model, and evaluating its performance.

## **Dataset**
The dataset used in this project is dvilasuero/ag_news_error_analysis, a dataset designed for error analysis on the AG News dataset. The dataset consists of text inputs and their corresponding labels, which are used to train and evaluate the model.
-	Inputs: News article text.
-	Labels: Categorical annotations indicating the class of the news article, with four possible labels (0, 1, 2, 3).
-	
### **Model**
* The model used for sequence classification is BERT (bert-base-uncased), a pre-trained transformer model from the Hugging Face library. BERT is particularly effective for text classification tasks due to its ability to understand context through bidirectional training.
* 
## **Preprocessing**
### **Tokenization**
The text data is tokenized using the BertTokenizer. Tokenization involves converting text into a format that the BERT model can process, including splitting text into tokens and converting them into corresponding IDs.
* Tokenizer: BertTokenizer.from_pretrained('bert-base-uncased')
* Max Token Length: 256
The tokenization process also includes padding to ensure all sequences have the same length and truncation for sequences longer than the maximum length.

## **Data Formatting**
After tokenization, the dataset is formatted into tensors suitable for model training. The dataset is also split into training and validation sets.
•	Training Set: 90% of the original training dataset.
•	Validation Set: 10% of the original training dataset.

### **Training**
The BERT model is fine-tuned on the tokenized dataset using the Trainer API from the Hugging Face library. The training involves optimizing the model's parameters to minimize the classification error on the training set.

#### **Training Configuration**
•	Optimizer: AdamW with weight decay
•	Learning Rate: 2e-5
•	Batch Size: 8 for both training and evaluation
•	Epochs: 3
•	Evaluation Strategy: Evaluation at the end of each epoch

# **Training Process**
The training process involves iterating over the training dataset multiple times (epochs) and updating the model's weights to improve classification accuracy.

# **Evaluation**
After training, the model's performance is evaluated on the validation set. The following metrics and methods are used for evaluation:
# **Metrics**
-	**Accuracy**: The proportion of correctly classified instances.
-	**Confusion Matrix**: A table used to describe the performance of the classification model, showing the true vs. predicted labels.
-	**Classification Report**: Includes precision, recall, F1-score for each class.
# **Visualization**
- **Confusion Matrix**: Visualized using Seaborn to provide insights into the classification performance.
# **Predictions**
In addition to evaluation, the trained model is used to classify a set of specific sentences. The predictions are compared with the true labels to assess the model's accuracy on unseen data.
# **Results**
The final results include the accuracy of the model on the validation set, a confusion matrix visualizing the model's performance, and a classification report detailing precision, recall, and F1-score for each class.
# **Summary**
This project demonstrates the effectiveness of BERT in sequence classification tasks. By fine-tuning BERT on a specific dataset, the model can achieve high accuracy and provide detailed insights into its classification performance.
