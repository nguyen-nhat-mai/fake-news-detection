# FAKE NEWS DETECTION
This project was done as part of the course Machine Learning course at CentraleSupelec.

## Project description
The task was to employ traditional machine learning (ML) classification methods to mine the text and identify unreliable news. The [dataset](https://www.kaggle.com/competitions/fake-news/data?select=train.csv) comprises 26,000 rows representing fake and reliable news. For each news, article title, text,
author and label are provided. ML methods used are Support vector machine (SVM), Naive bayes, Logistics regression, Knn, Decision tree. The performances of models were evaluated using accuracy score and running time. xx proved to outperform in this case with xx accuracy score while xx running time.

## Methodology
### 1. Pre-processing and explore data
2 issues found within this dataset:
1. Null values exist in title, author & text. Since there is no row with null values for all 3 columns, combing words from 3 columns is still valid for text analysis.

![image](https://user-images.githubusercontent.com/85484281/214556699-f85537fe-9cf3-43cf-b52e-8c6bdf804656.png)

2. 5 articles contain non-english contents. These were removed considering they are minority.

![image](https://user-images.githubusercontent.com/85484281/214557069-a0f2bad3-9120-4701-9448-789ae86700c3.png)

The text from title, author & text were combined and processed as followed:
1. Contraction (eg. "I'll" => "I will")
2. Lowercase
3. Remove punctuation and digits
4. Stemming (eg. "computing", "computer", "compute" => "compute")
5. Remove stopwords

Below is the snapshot of data after processing:

![image](https://user-images.githubusercontent.com/85484281/214557838-573d0436-e2f2-4862-b19d-05c779df92ba.png)
The distribution of fake and real news are balanced, thus requiring no further adjustment

![image](https://user-images.githubusercontent.com/85484281/214557893-4d063539-8474-445b-963d-32d96b3d135b.png)

Interestingly, "trump" appears frequently in true news while "hilarri clinton" appears frequently in fake news. (noted that the the word is a little bit different from the original word because of stemming)

![image](https://user-images.githubusercontent.com/85484281/214558607-64ade671-ce13-40a4-9926-8c5978c94e57.png)
![image](https://user-images.githubusercontent.com/85484281/214558631-78081ba8-33f0-4071-ba93-282e80acb84d.png)

### 2. Modelling & evaluation metrics:
Traditional ML classification models were imported from scikit-learn:
1. SVM
2. Naive Bayes
3. Knn
4. Logistics regression
5. Decision tree

The default models are tried first, then the hyperparameters are tuned to improve the performances. Cross-validation 5-fold was also used to prevent overfitting. For evaluation, following metrics are used:

1. [Accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) = # matches/ # sample
2. [Running time](https://docs.python.org/3/library/time.html) = time from fitting data into the model until finishing predicting the outcomes
3. [AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) = probability that the model ranks a random 1 example more highly than a random 0 example

### 4. Result
Model xxx achieved the best accuracy score but model xxx consumed the least tiem to run
=> table

Model xxx has the highest AUC
=> plot

Limitations in this project:
- Huge dimensions (

### 5. References:
https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae
https://medium.com/@pushkarmandot/what-is-the-significance-of-c-value-in-support-vector-machine-28224e852c5a
