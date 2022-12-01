# FAKE NEWS DETECTION
> You can run all the files in google colab. You can also find the  output results within those files. However, at the end of this file there is a comparision image of all the classifier results and the better results for some classifiers after hyper tuning the parameters.
### Introduction

###### <p align="justify">The internet has transformed the lives of countless people for the better or for the worse over the years.As the internet’s capabilities grew, so did the numberof legal and unlawful activities that it enabled. This communication channel might be used for illicit actions including sending threatening or abusive messages. By keeping an eye on their everyday posts and monitoring questionable topics, you can determine how loyal members are. We compared the accuracy of different machine learning algorithms which can analyze online plain text sources from the dataset and classify them into positive or negative news. First, we will use TF-IDF vectorizer which will convert a collection of raw documents to a matrix of TF-IDF features. This is a very common algorithm to transform text into a meaningful representation of numbers which is used to fit machine algorithms for prediction. We used K-Nearest Neighbour, Decision Tree, Random Forest, Naive-Bayes Logistic Regression, Passive Aggressive algorithm for our dataset.</p>

### Methodology

######  <p align="justify"> Data from the file was extracted into a dataframe. Data from this dataframe was pre-processed for missing or null values. This preprocessed data was then split into test and train data with 20% test data. After splitting data, features from the data were extracted with the help of TF-IDF vectorization. This extracted features were then fed to different classifiers for prediction. News were classified on different classifiers such as Decision Tree, Random Forest, Naive Bayes, Logistic Regression, KNN, Passive Aggressive and Linear SVC. Below image shows the steps followed to classify news. </p>

 <p align="center"><img src="/flow.png" width="650px" ></p>

**Data Preprocessing:**

######  <p align="justify">  It is very common for the dataset to have missing values. It may have occurred during data processing or as a result of a data validation rule, but missing values must be considered regardless. These missing values either are needed to be removed or filled otherwise these entries will cause errors. The ’news’ dataset we used was extracted into a data frame with the help of pandas library. Then we checked the total number of null entries in the dataset. There were no null entries in our dataset in all columns. Then we checked if there is any class imbalance in our dataset or not. But in our dataset there were 3171 entries for the class ”REAL” and 3164 entries for the class ”FAKE”. So there is no class imbalance in our dataset. Now Machine learning algorithms require numerical input and output variables. But our dataset has text data in columns so that we have to convert it into numerical data. For that we used TF-IDF vectorizor to convert it into TF-IDF features.</p>

**Data Splitting:**

######  <p align="justify"> As our dataset contains only 6335 data entries so we choose to take 80% of this data for training and the rest for testing purpose. Data was split with the help of train test split method of sklearn module. </p>

**Feature Extraction:**

######  <p align="justify"> Reducing Dimensionality is broad concept in machine learning. Reducing dimensions means selecting some features according to their importance towards class variable. To reduce dimensions there are two methods. 1) Feature Selection 2) Feature Extraction. In our dataset, We have used Feature Extraction method. We have used Term Frequency and Term Frequency-Inverse Document Frequency to get important features from the data. </p>

- ######  <p align="justify"> Term Frequency-Inverse Document Frequency: It is one of the most essential approaches for representing how relevant a certain word or phrase is to a given document in terms of information retrieval. The TF-IDF value rises in proportion to the number of times a word appears in the document, but is often countered by the word’s frequency in the corpus, which helps to compensate for the fact that some words appear more frequently than others in general. TF-IDF use two statistical methods, first is Term Frequency and the other is Inverse Document Frequency.The total number of times a specific term t appears in the document doc versus (per) the entire number of all words in the document is referred to as term frequency. The inverse document frequency is a measure of how much data a word contains. It counts the number of times a word appears in a document. IDF calculates the frequency of a term across all documents. </p>

**Classifiers:**

######  <p align="justify"> Classifiers are the algorithms that evaluates the given data and predict the required result[5]. First We have used KNN classifier to predict the outcome. But we didn’t get much accuracy using this classifier So we plan to implement more classifiers and to compare accuracy between them. </p>

- **K-Nearest Neighbour:**

######  <p align="justify"> KNN is one of the simplest classification algorithms. Even with such simplicity, it can give highly competitive result. KNN can also be used for regression problems. KNN captures the idea of similarity (sometimes called distance, proximity, or closeness). For calculating distance different methods like Euclidean distance, Minkowski distance, hamming distance are used. In KNN classification, after finding nearest neighbors majority voting is done among nearest neighbors i.e., target variable from majority class is chosen as output for test instances. The main task of KNN algorithm is to find best K value. When K=1 then target variable of first nearest neighbor is directly assign to test instance. In this case, there are high chances for overfitting problem. When K=n, then it chooses majority class among all instances available for training. In this scenario, our model becomes more simpler So there are chances for underfitting problem. We can choose value of K manually putting K values and also, we can run hyperparameter tuning to find best value of K. </p>

- **Naive Bayes:**

######  <p align="justify">  Naive bayes is a classification technique based on bayes theorem with an assumption that features are conditionally independent to each other. In simple terms a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. Naive Bayes model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods. </p>

- **Decision Tree:**

######  <p align="justify"> Decision tree is a ML technique where dataset is split into smaller groups until each division is clean and pure, and data classification is decided by the type of data. The pruning procedure is used for the final building of the tree when it is fully developed to remove noise from the dataset. Each path starting from the root represents a sequence of data splitting until a Boolean outcome is obtained at the leaf node in a decision tree technique. Each path in the decision tree is a decision rule that can be easily converted into human or programming languages in real life.There are multiple techniques capable of making decisions but decision tree is more clear and understandable. </p>

- **Random Forest:**

######  <p align="justify"> A random forest classifier is a well known ensemble classification technique used in machine learning and data science for a variety of applications. This method uses ”parallel ensembling,” which includes fitting multiple decision tree classifiers in parallel on various data set sub-samples and determining the conclusion or final outcome using majority voting or averages. As a result, the over-fitting problem is reduced while forecast accuracy and control are improved. As a result, a multi-decision tree RF learning model is more accurate than a single-decision tree model. It uses bootstrapping with decision trees to create a succession of decision trees with controlled variation. </p>

- **Logistic Regression:**

######  <p align="justify"> Although the name indicates that it is regression algorithm. But Logistic Regression is a classification algorithm. As we are classifying text on the basis of a wide feature set, with a binary output (true/false or true article/fake article), a logistic regression (LR) model is used, since it provides the intuitive equation to classify problems into binary or multiple classes. As name suggests it uses logistic function called sigmoid function. The sigmoid function/logistic function is a function that resembles an “S” shaped curve when plotted on a graph. It takes values between 0 and 1 and “squishes” them towards the margins at the top and bottom, labeling them as 0 or 1. </p>

- **Passive-aggressive:**

######  <p align="justify"> The Passive-Aggressive algorithms are a subset of Machine Learning algorithms which is quite useful and efficient in some circumstances. Passive-Aggressive algorithms don’t require learning rate . They do, however, provide a parameter for regularisation. Algorithms that are referred described as ”passive-aggressive algorithms” include those that: Passive: If the forecast is correct, do not make any changes to the model. To put it another way, the data in the example is inadequate to cause any model changes. Aggressive: If the forecast is incorrect, be aggressive. </p>

- **Support Vector Machine:**

######  <p align="justify"> A Support Vector Machine (SVM) is a machine learning method that examines data and divides it into two categories. The goal of the SVM method is to determine the best line or decision boundary for categorising n-dimensional space so that fresh data points can be placed in the correct category easily in the future. The extreme points/vectors that contribute in the creation of the hyperplane are chosen using SVM. The Support Vector Machine is named after these extreme examples, which are referred to as support vectors. In our case it is a binary classification hence we used LinearSVC for classification. </p>

**Hyperparameter Tuning**

######  <p align="justify"> It is important for any machine learning algorithm to find best value of parameters to get the maximum accuracy according to dataset. To find the best value of parameters, there are 2 techniques available in scikit learn package. 1) GridSearchCV 2)RandomizedSearchCV. In our implementation we have used both this technique to find best model according to our dataset. GridSearchCV method is checking all permutation and combinations of parameters which is given to their parameters with 5 cross validation. From that we can select best output for classifier. From RandomizedSearchCV, we can find best classifier for our dataset as we can pass different models with range of different parameters in argument. </p>

######  <p align="justify"> To improve the performance of the classifiers we did hyper-parameter tuning with GridSearch and RandomizedSearch. With Randomized search performance of the model wasn’t improved so we tried GridSearch. Grid Search hyperparameter tuning increased the performance of KNN model from 50.35% to 65.35%. We did Grid Search tuning on Decision tree with different depths up to 30. In GridSeachCV ,for decision tree classifier we get accuracy show in following table. From table we can conclude that, as we increase depth of decision tree accuracy increases at certain point. But when we select most of features, accuracy start decreasing. Same things happened in case of Random Forest classifier. As we increase number of estimators accuracy increases slightly. We get the highest accuracy of 83% for our dataset. Following Image shows the comparision of the accuracies among different classifiers. </p>

 <p align="center"><img src="/comparision.PNG" width="650px" ></p>
