# FAKE NEWS DETECTION
This project was done as part of the Machine Learning course at CentraleSupelec.

## Project description
The task was to employ traditional machine learning (ML) classification methods to mine the text and identify unreliable news. The [dataset](https://www.kaggle.com/competitions/fake-news/data?select=train.csv) comprises 26,000 rows representing fake and reliable news. For each news, article title, text,
author and label are provided. ML methods used are Support vector machine (SVM), Naive bayes, Logistics regression, Knn, Decision tree. The performances of models were evaluated using accuracy score and running time. xx proved to outperform in this case with xx accuracy score while xx running time.

## Methodology
### 1. Pre-processing and explore data
2 issues were found in this dataset:
- Null values exist in title, author & text. Since there is no row with null values for all 3 columns, combing words from 3 columns is still valid for text analysis.

![image](https://user-images.githubusercontent.com/85484281/214556699-f85537fe-9cf3-43cf-b52e-8c6bdf804656.png)

- 5 articles contain non-english contents. These were removed considering they are minority.

The text from title, author & text were combined and processed as followed:
- Contraction (eg. "I'll" => "I will")
- Lowercase
- Remove punctuation and digits
- Stemming (eg. "computing", "computer", "compute" => "compute")
- Remove stopwords

Below is the snapshot of data after processing:

![image](https://user-images.githubusercontent.com/85484281/214557838-573d0436-e2f2-4862-b19d-05c779df92ba.png)
The distribution of fake and real news are balanced, thus requiring no further adjustment

![image](https://user-images.githubusercontent.com/85484281/214557893-4d063539-8474-445b-963d-32d96b3d135b.png)

Interestingly, "trump" appears frequently in true news while "hilarri clinton" appears frequently in fake news. (noted that the the word is a little bit different from the original word because of stemming)

![image](https://user-images.githubusercontent.com/85484281/214558607-64ade671-ce13-40a4-9926-8c5978c94e57.png)
![image](https://user-images.githubusercontent.com/85484281/214558631-78081ba8-33f0-4071-ba93-282e80acb84d.png)

The combined text, finally, are vectorized using [TF-IDF](https://www.capitalone.com/tech/machine-learning/understanding-tf-idf/)

### 2. Modelling & evaluation metrics:
Traditional ML classification models were imported from scikit-learn:
- SVM
- Naive Bayes
- Knn
- Logistics regression
- Decision tree

The default models are tried first, then the hyperparameters are tuned to improve the performances. Cross-validation 5-fold was also used to prevent overfitting. For evaluation, following metrics are used:

- [Accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) = # matches/ # sample
- [Running time](https://docs.python.org/3/library/time.html) = time from fitting data into the model until finishing predicting the outcomes
- [AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) = probability that the model ranks a random 1 example more highly than a random 0 example

## Findings and some comments
### 1. Findings
Model xxx achieved the best accuracy score but model xxx consumed the least tiem to run
=> table

Model xxx has the highest AUC
=> plot
### 2. Limitations and suggetions
As concluded, the training data does not represent the population, thus causing the model to learn wrong behaviors and yield inaccurate output on test data. Since the method to collect the data is not revealed, it is difficult to handle this problem. A suggestion to make this project more applicable is to webscrap data from the social media and newspapers.

By vectorizing the text, huge dimensions (161,275 features in this case) can not be avoided. It is, therefore, difficult or impossible to run and tune hyper paramters of dimension-sensitive models such as SVM, Knn, Random forest, .... Techniques to reduce the dimension such as PCA & LDA could not be implemented because 1) the vectorized data are sparse matrix (mostly include 0) and 2) my computer memory (RAM) is limited



## References
https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae
https://medium.com/@pushkarmandot/what-is-the-significance-of-c-value-in-support-vector-machine-28224e852c5a
