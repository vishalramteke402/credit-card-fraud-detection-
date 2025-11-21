CREDIT CARD FRAUD DETECTION 💳
USING MACHINE LEARNING

Context
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

Content & Data Engineering
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset is characterized by a severe class imbalance:

Total Records: 284,807

Fraudulent Transactions (Class 1): 492

Fraud Rate: 0.172%

This presents a critical challenge, as most machine learning algorithms assume a balanced class distribution. We address this using undersampling during model training.

Feature Engineering
The data features are results of confidentiality measures:

Features V1, V2, ... V28: The principal components obtained through Principal Component Analysis (PCA). PCA is a dimensionality reduction technique used to extract the most important information from a high-dimensional dataset while preserving as much variance as possible.


Shutterstock
Explore
'Time' and 'Amount': The only features not transformed by PCA. 'Amount' is normalized during preprocessing.

'Class': The target variable (1 for fraud, 0 for normal).

Inspiration
The primary goal is to Identify fraudulent credit card transactions.

Given the class imbalance, the standard metric of Confusion Matrix Accuracy is highly misleading. We recommend measuring performance using the Area Under the Precision-Recall Curve (AUPRC) and focusing on Precision and Recall for the minority (fraud) class.

Modeling and Prediction
The project explores both unsupervised anomaly detection methods and a supervised classification model, typically trained on an undersampled version of the data to mitigate class bias.

1. Unsupervised Anomaly Detection
Anomaly detection is used here because fraud patterns are non-stationary (constantly changing), making unsupervised methods potentially more robust and generalizable than supervised learning, which may overfit to past fraud patterns.

Isolation Forest (IF)
Isolation Forest is based on the idea of isolating anomalies instead of profiling normal data points. Anomalies are data points that are few and different, making them susceptible to isolation closer to the root of a random decision tree.

Local Outlier Factor (LOF)
LOF measures the local density deviation of a given data point with respect to its neighbors, flagging samples that have a substantially lower density than their neighbors as outliers.

One-Class SVM (OCSVM)
OCSVM learns a decision boundary around the "normal" data points, separating them from the origin in the feature space. Any new data point falling outside this boundary is classified as an anomaly.

2. Supervised Classification
Random Forest (RF)
A standard Random Forest classifier was trained on the undersampled data for benchmarking against the unsupervised methods.

Results and Evaluation
The models were evaluated primarily on their ability to correctly identify the minority class.

Classifier	Accuracy	Precision (Fraud)	Recall (Fraud)
Isolation Forest (Unsupervised)	92.4%	85%	100%
Local Outlier Factor (LOF) (Unsupervised)	60%	50%	72%
One-Class SVM (Unsupervised)	88%	78%	100%
Random Forest (Supervised)	99.8%	83%	79%

Export to Sheets

Key Metrics Analysis
Recall (Sensitivity): The percentage of actual fraud cases correctly identified (True Positives / All Actual Positives). A high recall is often paramount in fraud detection to minimize costly undetected fraud.

Precision (Positive Predictive Value): The percentage of predicted fraud cases that were actually fraud (True Positives / All Predicted Positives). Low precision means many legitimate transactions are flagged, leading to false alarms and customer inconvenience.

Conclusion
Isolation Forest and One-Class SVM both achieved perfect Recall (≈100%) on the undersampled test set, demonstrating excellent ability to catch all fraud attempts. The Isolation Forest model, however, had better precision (85% vs. 78%), making it the superior unsupervised choice for minimizing false positives while maintaining high fraud capture.

The Random Forest model achieved the highest raw accuracy (99.8% on the original test set), but its performance on the fraud class (Recall 79%) suggests it missed more fraud cases compared to the best unsupervised models. This validates the use of unsupervised methods when tackling evolving, non-stationary fraud patterns.
