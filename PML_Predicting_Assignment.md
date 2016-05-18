# PML Predicting Assignment
Mahbub khan  
May 16, 2016  

##Overview
In this Coursera Practical Machine Learning Predicting Assignment, a Human Activity Recognition (HAR) dataset (http://groupware.les.inf.puc-rio.br/har) is analysed, where six participants completed some weight lifting exercise whilst being monitored, the study focused on measuring how well they did the exercise instead of conventional focus of measuring how much.The study quantified a catergorical variable called "classe" with five levels, this assignment aims to predict the classe variable as the outcome using machine learing algorithms learnt from the course.
Firstly, data is downloaded from the website, saved into home drive and testing and training csv files loaded onto R.Afterwards, test data was cleaned and correlations between variables was checked to find out any pattern and suitability of fitting regression models. Failing to find any significant correlation, tree based classification models were used to predict the outcome. Firstly, a Recursive Partitioning And Regression Trees (RPART) model was fit, with in sample accuracy only 50%, then a Random Forest model was fit with model accuracy 99.35%.


##Data Collection & Loading

Data downloaded and training and test csv files loaded onto R


```r
#URL's of training and test dataset
trainurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

testurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

#Saving csv files onto home drive's data folder
if (!file.exists("./data/pml-training.csv")){
  download.file(trainurl, destfile="./data/pml-training.csv", method="curl")}


if (!file.exists("./data/pml-testing.csv")){
download.file(testurl, destfile="./data/pml-testing.csv", method="curl")
}


library(caret)# Loading caret package
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
#Loading trainig and test data
pmltraindata<-read.csv("./data/pml-training.csv",header=TRUE)
pmltest20<-read.csv("./data/pml-testing.csv",header = TRUE)
```

##Data Cleaning

In order to clean data, firstly columns with missing values are deleted.

```r
#Removing columns with missing values
trnacln<-pmltraindata[,!sapply(pmltraindata,function(x) any(is.na(x)))]
tstnacln<-pmltest20[,!sapply(pmltest20,function(x) any(is.na(x)))]
```

Afterwards, coulumns with nonzero variance were deleted.

```r
#Removing variables with near zero variance
nzv <- nearZeroVar(trnacln, saveMetrics=TRUE)
nzvclntrain <- trnacln[, nzv$nzv==FALSE]
nzvt <- nearZeroVar(tstnacln, saveMetrics=TRUE)
nzvtstcln <- tstnacln[, nzvt$nzv==FALSE]
```

Lastly, first six columns of training dataframe was deleted as they bear no significance to prediction.

```r
#Removing first 6 columns for persimony
finalclntrain<-nzvclntrain[,-c(1:6)]
clnpmltest20<-nzvtstcln[,-c(1:6)]
```

##Cross Validation

For cross validation, the training data was divided in to training and testing data using caret packages createDataPartion function, and training data is used for model training and test data for validation.


```r
#Cross valiating 
inTraining <- createDataPartition(finalclntrain$classe, p = .75, list=FALSE)
training <- finalclntrain[inTraining,]
testing <- finalclntrain[-inTraining,]
```


##Exploratory Analysis
In order to check correlations of classe with other 160 variables, the variables were treated as numeric using as.numeric function, and their correlations was obtained using corr function. Then correlation table was turned into dataframe and checked for the correlations of classe with all other variables with a threshold of .4. 


```r
#Creating a numeric "classe"" variable and then making other var's numeric to #check correlations 
numtraining<-training
numtraining$classe<-as.numeric(numtraining$classe)#turning classe to numeric
cortbl<-cor(numtraining[,unlist(lapply(numtraining, is.numeric))])
cordf<-as.data.frame(cortbl)#making table into data frame
cordf[,53]>.4 | cordf[,53]<(-.4) # checking for positive or negetive #correlation greater than .4, classe is 53rd column
```

```
##  [1] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [12] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [23] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [34] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [45] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE  TRUE
```

From, the R output we can observe only one the last output is TRUE, which is the correlation of classe with itself. From our exploration, we decide not to use any regression model and try classification tree model.


##Model Fitting & Evaluation


###CART Model

Firstly an rpart model is fit and model accuracy is checked

```r
treefit<-train(classe~.,data=training,method="rpart")# fitting a tree model
```

```
## Loading required package: rpart
```

```r
treefit
```

```
## CART 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.03284914  0.5139236  0.36756514  0.02259559   0.03444229
##   0.05952720  0.4022742  0.18923122  0.06089103   0.10025192
##   0.11411754  0.3392395  0.08712815  0.03794081   0.05551027
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03284914.
```

The model accuracy is 50%, which is not satisfoactory.

###Random Forest Model

Afterwards, we fit a random forest model, which as sophisticated model based on classification tree with repeated and resampled fitting of tree models with a view to improving on accuracy. As random forst is very complex procedure and demands a lot of processing time, parallel processing is best option to improve runtime performance. We used guidance form Mr. L Greski (https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md) to implement parallel processing.


```r
x <- training[,-53]
y <- training[,53]
library(parallel)
library(doParallel)
```

```
## Warning: package 'doParallel' was built under R version 3.2.5
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

```r
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = TRUE)
rffit <- train(x,y, method="rf",data=training,trControl = fitControl)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.2.5
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
stopCluster(cluster)
rffit
```

```
## Random Forest 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 13245, 13245, 13246, 13246, 13246, 13247, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9924584  0.9904594  0.001614559  0.002043848
##   27    0.9928659  0.9909753  0.002921591  0.003696372
##   52    0.9864112  0.9828110  0.003343505  0.004229678
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

We obtain a model accuracy of 99.20% and decide to use this model for prediction and measuring out of sample error.


##Prediction

Using our random forest model we predict using our validation data set and print the confusion matrix.


```r
rfpredict<-predict(rffit,newdata=testing)#using random forest model and validation data
confusionMatrix(rfpredict, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    6    0    0    0
##          B    1  942    5    0    1
##          C    0    1  849    6    4
##          D    0    0    1  795    3
##          E    1    0    0    3  893
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9935          
##                  95% CI : (0.9908, 0.9955)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9917          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9986   0.9926   0.9930   0.9888   0.9911
## Specificity            0.9983   0.9982   0.9973   0.9990   0.9990
## Pos Pred Value         0.9957   0.9926   0.9872   0.9950   0.9955
## Neg Pred Value         0.9994   0.9982   0.9985   0.9978   0.9980
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2841   0.1921   0.1731   0.1621   0.1821
## Detection Prevalence   0.2853   0.1935   0.1754   0.1629   0.1829
## Balanced Accuracy      0.9984   0.9954   0.9951   0.9939   0.9951
```

From the confusion matrix we see that out of sample erro rate is (1-.9935) 0.0065

Finally, we use the 20 observations of test data to predict 20 classe predictions.


```r
predict (rffit ,  newdata=clnpmltest20)#using rf model to predic 20 test classe values
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


## Conclusion
In conclusion we can coment that, random forest is a very efficent model for classification but it takes plenty of processing time and for that parallel processing is alway advised.



