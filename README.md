# FAKE NEWS DETECTION
This project was done as part of the course Machine Learning course at CentraleSupelec.

## Project description
The task was to employ traditional machine learning (ML) classification methods to mine the text and identify unreliable news. The [dataset](https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php) comprises 44,898 rows representing fake and reliable news. For each news, article title, text,
type and the date the article was published on are provided. The majority of the news focus on political and world news topics. ML methods employed are Naive Bayes, Logistics Regression, Knn, Decision Tree. The performances of models were evaluated using accuracy score and running time. Decision Tree proved to outperform in this case.

## Methodology
### 1. Pre-processing and explore data
The text from title, text, subject were combined and processed as followed:
1. Contraction (eg. "I'll" => "I will")
2. Lowercase
3. Remove punctuation and digits
4. Stemming (eg. "computing", "computer", "compute" => "compute")
5. Remove stopwords

Below is the snapshot of data after processing:

![image](https://user-images.githubusercontent.com/85484281/212895836-c78b4fb1-828f-4ef6-a43e-a1a299269ea1.png)
The distribution of fake and real news are balanced, thus requiring no further adjustment

![image](https://user-images.githubusercontent.com/85484281/212898337-e1b65609-c196-4f17-b456-35ca155ed292.png)
Interestingly, "donald trump" appears frequently in both real and fake news. Fake news frequently include words such as "one", "said", "call" while "unit state", "white hous" are common in real news. (noted that the the word is a little bit different from the original word because of stemming)

![image](https://user-images.githubusercontent.com/85484281/212899123-6cbc7fb5-c90c-44e0-ad78-f9f9885b2788.png)
![image](https://user-images.githubusercontent.com/85484281/212899162-4fab42da-46ef-473e-8244-43eb8b17cf2d.png)

### 2. Modelling
Traditional ML classification models were imported from scikit-learn:
1. Naive Bayes
2. Knn
3. Logistics regression
4. Decision tree

The default models are tried first, then the hyperparameters are tuned to improve the performances. Cross-validation 5-fold was also used to prevent overfitting.
### 3. Evaluation
[Accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) = # matches/ # sample

[Running time](https://docs.python.org/3/library/time.html) = time from fitting data into the model until finishing predicting the outcomes

[AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) = probability that the model ranks a random 1 example more highly than a random 0 example
### 4. Result
Model xxx achieved the best accuracy score but model xxx consumed the least tiem to run
=> table

Model xxx has the highest AUC
=> plot
