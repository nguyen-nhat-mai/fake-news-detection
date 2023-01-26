# FAKE NEWS DETECTION
This project was done as part of the Machine Learning course at CentraleSupelec.

## Project description
The task was to employ traditional machine learning (ML) classification methods to mine the text and identify unreliable news. The [dataset](https://www.kaggle.com/competitions/fake-news/data?select=train.csv) comprises 26,000 rows representing fake and reliable news. For each article, title, text,
author and label are provided. ML methods employed are Support vector machine (SVM), Naive bayes, Logistics regression, Knn and Decision tree. The performances of models were evaluated using accuracy score and running time. Naive Bayes proved to outperform in this case with 84% accuracy score and the least running time i.e. 0.11 seconds.

## Methodology
#### 1. Pre-processing and explore data
2 issues were found in this dataset:
- Null values exist in title, author & text. Since there is no row with null values in all 3 columns, combining words from 3 columns is valid for text analysis.

![image](https://user-images.githubusercontent.com/85484281/214839883-058fabf0-5b61-44cd-b908-9810bc9fbeac.png)

- 5 articles contain non-english contents. These were removed considering they are minority.

The text from title, author & text were combined and processed as followed:
- Contraction (eg. "I'll" => "I will")
- Lowercase
- Remove punctuation and digits
- Stemming (eg. "computing", "computer", "compute" => "compute")
- Remove stopwords

Below is the snapshot of data after processing:

![image](https://user-images.githubusercontent.com/85484281/214839720-5a808184-a7ae-418f-9fd8-d054812a4592.png)

The distribution of fake and real news are balanced, thus requiring no further adjustment. Interestingly, "trump" appears frequently in true news while "hilarri clinton" appears frequently in fake news. (noted that the the word is a little bit different from the original word because of stemming)

![image](https://user-images.githubusercontent.com/85484281/214839441-b4b19b9a-31ff-4b30-823a-7e7a15ed8f66.png)

Finally, the combined text are vectorized using [TF-IDF](https://www.youtube.com/watch?v=vZAXpvHhQow) - a statistical method to measure how relevant a word is. Intuitively, words that either do not appear in an article or appear in all articles are not important (assigned value is 0). 

![image](https://user-images.githubusercontent.com/85484281/214850138-ea8debf8-f01b-4214-836d-365c9460339c.png)

#### 2. Modelling & evaluation metrics:
Traditional ML classification models were imported from scikit-learn: SVM, Naive Bayes, Knn, Logistics regression, Decision tree. The default models are tried first, then the hyperparameters are tuned using GridSearchCV to improve the performances. Cross-validation 5-fold was also used to prevent overfitting. For evaluation, following metrics are used:

- [Accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) = # matches/ # sample
- [Running time](https://docs.python.org/3/library/time.html) = time from fitting data into the model until finishing predicting the outcomes

## Findings
#### 1. Result
Bernoulli Naive Bayes achieved the best accuracy score with 84% accuracy score and the least running time i.e. 0.11 seconds

![image](https://user-images.githubusercontent.com/85484281/214859074-eee02fe1-ab3b-455d-92ea-5d86cd2b9e3b.png)

Except for Naive Bayes, the accuracy scores are extremely high when models are trained on training dataset (91-100%) but drop significantly when applied upon the test dataset (58-64%) (noted that train-test set split is fixed following the rule of the Kaggle competition). The models are significantly overfitting and the training dataset seems to not representing the population well. As a result, the hyperparameters obtained after 5-fold cross-validating on training dataset are not valid as well. On such ground, I focused on adding regularization to prevent overfitting and managed to improve the performance of SVM (70%) and Logistics regression (70%)

![image](https://user-images.githubusercontent.com/85484281/214859621-5a755722-b8c8-4f03-a682-a5173572b4ea.png)

This method, however, does not work for Knn & Decision tree.Training scores are closed to perfect (100%) while validation scores remain the same no matter how the parameters are set.

![image](https://user-images.githubusercontent.com/85484281/214862638-1317a7f3-c8e8-495e-8dec-326460e82de1.png)

#### 2. Limitations and suggetions
As concluded, the training data does not represent the population, thus causing the model to learn wrong behaviors and yield inaccurate output on test data. Since the method to collect the data is not revealed, it is difficult to handle this problem. A suggestion to make this project more applicable is to webscrap data from the social media and newspapers.

By vectorizing the text, huge dimensions (161,275 features in this case) can not be avoided. It is, therefore, difficult or impossible to run and tune hyper paramters of dimension-sensitive models such as SVM, Knn, Decision Tree, .... Techniques to reduce the dimension such as PCA & LDA could not be implemented because 1) the vectorized data are sparse matrix (mostly include 0) and 2) my computer memory (RAM) is limited

## References
https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae
https://medium.com/@pushkarmandot/what-is-the-significance-of-c-value-in-support-vector-machine-28224e852c5a
