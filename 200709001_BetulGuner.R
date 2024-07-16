#importing the libraries
library(ggplot2) 
library(ggthemes)
library(plotly)
library(dplyr)
library(psych)
install.packages("corrplot")
library(corrplot)
install.packages("corrr")
library(corrr)
library(tidyverse)
install.packages(caret)
library(caret)
library(lattice)
library(class)
install.packages("rpart")
library(rpart)
library(e1071)
install.packages("randomForest")
library(randomForest)
install.packages("MASS")
library(MASS)
install.packages("psych")
library(psych)


#Examine and understand the data set
data <- read.csv('/Users/betulguner/Documents/200709001_BetulGuner/hearth.csv') #read the data folder
head(data) #get the first 5 data from the data set

summary(data) #it gives the statistical infromations about the data set
is.data.frame(data) #controls is it a data set or not
ncol(data) #number of column
nrow(data) #number of row
dim(data) #dimensions
names(data) #gives the column names

#-----------------------------------------------------------------------------------------------
#VISUALISATION PART
#***
install.packages("corrplot")
library(corrplot)
M <- cor(data[, -13])  # Create correlation matrix
corrplot(M, method = "color")  #Draw a correlation chart

#***
png(file = "columnsRelation.png") #it saves the .png format of the graph
pairs(data) #it shows the relationships between all columns as a matrix
dev.off()

#***
for (i in names(data))  #bar graph
{
  if (is.numeric(data[[i]]) && i!="target") {  #If the column is numeric and not the target column
    p <- ggplot(data = data, aes_string(x = i, y = "target")) +
      geom_bar(stat = "identity") +
      ggtitle(paste("Bar Plot of", i))
    print(p)
    Sys.sleep(2) #wait 2 seconds
  }
}

#***
for (col in names(data)) # density graph
{
  if (is.numeric(data[[col]]) && col != "target") {  # Eğer sütun numerik ise ve target sütunu değilse
    p <- ggplot(data = data, aes_string(x = col)) +
      geom_density() +
      ggtitle(paste("Density Plot of", col))
    print(p)
    Sys.sleep(2) #wait 2 seconds
  }
}
#end of the visualization part

#PRE-PROCESSING
#check is there any null value
print(is.na(data))
print(sum(is.na(data))) 

#create the data set with missing values
set.seed(123)
heart_with_missing <- data
heart_with_missing[sample(1:nrow(heart_with_missing),2),"thalach"] <- NA 
heart_with_missing[sample(1:nrow(heart_with_missing),1),"oldpeak"] <- NA
print(sum(is.na(heart_with_missing)))

#fill the missing values by the mean value of the thalach column
heart_with_missing$thalach <- 
  ifelse(is.na(heart_with_missing$thalach),
         mean(heart_with_missing$thalach, na.rm = TRUE),
         heart_with_missing$thalach)

#print(sum(is.na(heart_with_missing))) #check is it filled or not

#fill the missing values by the mean value of the oldpeak column
heart_with_missing$oldpeak <-  
  ifelse(is.na(heart_with_missing$oldpeak),
         mean(heart_with_missing$oldpeak, na.rm = TRUE),
         heart_with_missing$oldpeak)

#print(sum(is.na(heart_with_missing))) #check is it filled or not


#-----------------------------------------------------------------------------------------------
#OUTLIER DATA:
#visualization of outlier data
boxplot(heart_with_missing[,1:13])

#IQR Method
dataSet <- heart_with_missing[, 1:13]

# Finding the outlier data function for each column
find_outliers <- function(column) 
{
  Q1 <- quantile(column, 0.25)
  Q3 <- quantile(column, 0.75)
  IQR_value <- Q3 - Q1
  
  min <- Q1 - 1.5 * IQR_value
  max <- Q3 + 1.5 * IQR_value
  
  outliers <- column[(column < min) | (column > max)]
  return(outliers)
}

#Function of replace outliers with the max non-outlier data value in that column
changeOutlier <- function(column)
{
  Q1 <- quantile(column, 0.25)
  Q3 <- quantile(column, 0.75)
  IQR_value <- Q3 - Q1
  
  min <- Q1 - 1.5 * IQR_value
  max <- Q3 + 1.5 * IQR_value
  
  non_outliers <- column[(column >= min) & column <=max]
  max_non_outlier <- max(non_outliers, na.rm=TRUE)
  column[(column < min) | (column>max)] <- max_non_outlier
  
  return(column)
}

#Finding the outlier data for each column
outliers_list <- lapply(dataSet, find_outliers)
print(outliers_list)

#number of outlier data for each column
print(outlier_counts <- sapply(outliers_list, length))

#replacing the outlier data with the max non-outlier data
max_non_outliers <- sapply(heart_with_missing[,1:13],changeOutlier) 
#print(max_non_outliers)

#-----------------------------------------------------------------------------------------------
#CLASSIFICATION(6 types)
# 1) KNN: Apply the KNN algorithm on the data set WITHOUT normalization 
#split the data set into train(70%),test(15%), and validation(15%)
set.seed(123)
trainIndex <- createDataPartition(data$target, p = 0.7, list = FALSE)
trainDataa <- data[trainIndex,]
temporaryData <- data[-trainIndex,]

validationIndex <- createDataPartition(temporaryData$target, p = 0.5, list = FALSE)
validationDataa <- temporaryData[validationIndex,]
testDataa <- temporaryData[-validationIndex,] #real values

#check the number of rows of the splitted data
nrow(trainDataa) 
nrow(testDataa)
nrow(validationDataa)

library(caret)
library(lattice)
library(class)

#apply the knn algorithm
knn_modell <- knn(train = trainDataa[,-13], 
                  test = testDataa[,-13],
                  cl = trainDataa$target,
                  k = 7)
knn_modell_val <- knn(train = trainDataa[,-13], 
                  test = validationDataa[,-13],
                  cl = trainDataa$target,
                  k = 7)

knn_modell #predicted test values
knn_modell_val #predicted validation values
print(c(testDataa$target)) #real values

#find the confusion matrix
knn_confusion_matrixxx <- table(Reference = testDataa$target,  
                            Prediction = knn_modell)
cat("Confusion matrix of test data:")
knn_confusion_matrixxx

knn_confusion_matrix_val <- table(Reference = testDataa$target,  
                            Prediction = knn_modell_val)
cat("Confusion matrix of test data:")
knn_confusion_matrix_val

#calculate the accuracy of original data set
knn_accuracy_originalll <- sum(diag(knn_confusion_matrixxx))/sum(knn_confusion_matrixxx)
cat("Accuracy for original data:",knn_accuracy_originalll)

knn_accuracy_original_val <- sum(diag(knn_confusion_matrix_val))/sum(knn_confusion_matrix_val)
cat("Accuracy for original data:",knn_accuracy_original_val)

#sensitivity
TP <- knn_confusion_matrixxx[2,2] #number of true positive values
FN <- knn_confusion_matrixxx[2,1] #number of false negative values

knn_sensitivity <- TP / (TP + FN) #sensitivity calculation
print(knn_sensitivity)

TP <- knn_confusion_matrix_val[2,2] #number of true positive values
FN <- knn_confusion_matrix_val[2,1] #number of false negative values

knn_sensitivity_val <- TP / (TP + FN) #sensitivity calculation
print(knn_sensitivity_val)

#specificity
TN <- knn_confusion_matrixxx[1,1] #number of true negative values 
FP <- knn_confusion_matrixxx[1,2] #number of false positive values

knn_specificity <- TN / (TN + FP) #specificity calculation
print(knn_specificity)

TN <- knn_confusion_matrix_val[1,1] #number of true negative values 
FP <- knn_confusion_matrix_val[1,2] #number of false positive values

knn_specificity_val <- TN / (TN + FP) #specificity calculation
print(knn_specificity_val)

#F1-Score
TP <- knn_confusion_matrixxx[2,2] #number of true positive values
FP <- knn_confusion_matrixxx[1,2] #number of false positive values
FN <- knn_confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

knn_f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(f1_score)

TP <- knn_confusion_matrix_val[2,2] #number of true positive values
FP <- knn_confusion_matrix_val[1,2] #number of false positive values
FN <- knn_confusion_matrix_val[2,1] #number of false negative values
precision_val <- TP / (TP + FP)
recall_val <- TP / (TP + FN)

knn_f1_score_val <- (2 * (precision_val + recall_val)) / (precision_val + recall_val) #f1_score calculation
print(knn_f1_score_val)

# #results of KNN classification visualization:
# Storing metric values in variables as vectors
knn_accuracy <- c(knn_accuracy_originalll, knn_accuracy_original_val)
knn_sensitivity <- c(knn_sensitivity, knn_sensitivity_val)
knn_specificity <- c(knn_specificity, knn_specificity_val)
knn_f1_scores <- c(knn_f1_score, knn_f1_score_val)

# Creating a dataframe
metrics_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Test = c(knn_accuracy[1], knn_sensitivity[1], knn_specificity[1], knn_f1_scores[1]),
  Validation = c(knn_accuracy[2], knn_sensitivity[2], knn_specificity[2], knn_f1_scores[2])
)

# Plotting bar plots for comparison
par(mfrow = c(2, 2))  # Setting the layout for multiple plots

metrics_df_long <- reshape2::melt(metrics_df, id.vars = "Metric")

# Bar plot for Accuracy, Sensitivity, Specificity, and F1 Score
ggplot(metrics_df_long, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.7) +
  labs(title = "Metrics Comparison",
       x = "Metrics",
       y = "Values",
       fill = "Dataset") +
  scale_fill_manual(values = c("skyblue", "salmon")) +
  geom_text(aes(label = round(value, 3)), position = position_dodge(width = 0.9), vjust = -0.25) +
  theme_minimal()

library(gplots)
plot(as.table(knn_confusion_matrixxx), main = "Confusion Matrix")
plot(as.table(knn_confusion_matrix_val), main = "Confusion Matrix Validation")


# 2)LOGISTIC REGRESSION: Apply the logistic regression on the data set WITHOUT normalization
independent_var <- heart_with_missing[, 1:13]
dependent_var <- heart_with_missing[, 14]

logit_model <- glm(dependent_var ~ ., data = heart_with_missing, family = binomial) #compose the logistic regression model
logit_model

lr_predictions <- predict(logit_model, newdata = heart_with_missing, type="response")
lr_predicted_classes <- ifelse(predictions > 0.5,1,0)

lr_pred_val <- predict(logit_model, newdata = validationDataa, type="response")
lr_predicted_classes <- ifelse(predictions > 0.5,1,0)

#find the confusion matrix
lr_confusion_matrixxx <- table(lr_predicted_classes,heart_with_missing$target)
print(lr_confusion_matrixxx)

lr_confusion_matrix_val <- table(lr_pred_val,validationDataa$target)
print(lr_confusion_matrix_val)

#calculate the accuracy of original data set
lr_accuracy_originalll <- lr_confusion_matrixxx['Accuracy']
print(paste("Accuracy:",lr_accuracy_originalll))

lr_accuracy_original_val <- lr_confusion_matrix_val['Accuracy validation']
print(paste("Accuracy:",lr_accuracy_original_val))

#sensitivity
lr_sensitivity <- lr_confusion_matrixxx['Sensitivity']
print(paste("Sensitivity",lr_sensitivity))

lr_sensitivity_val <- lr_confusion_matrix_val['Sensitivity validation']
print(paste("Sensitivity",lr_sensitivity_val))

#specificity
lr_specificity <- lr_confusion_matrixxx['Specificity']
print(paste("Specificity",lr_specificity))

lr_specificity_val <- lr_confusion_matrix_val['Specificity']
print(paste("Specificity",lr_specificity_val))

#F1-Score
lr_f1_score <- lr_confusion_matrixxx['F1']
print(paste("F1 Score",lr_f1_score))

lr_f1_score_val <- lr_confusion_matrix_val['F1']
print(paste("F1 Score",lr_f1_score_val))

#results of Logistic Regression classification visualization:
# Storing metric values in variables as vectors
lr_accuracy <- c(lr_accuracy_originalll, lr_accuracy_original_val)
lr_sensitivity <- c(lr_sensitivity, lr_sensitivity_val)
lr_specificity <- c(lr_specificity, lr_specificity_val)
lr_f1_scores <- c(lr_f1_score, lr_f1_score_val)

# Creating a dataframe
metrics_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Test = c(lr_accuracy[1], lr_sensitivity[1], lr_specificity[1], lr_f1_scores[1]),
  Validation = c(lr_accuracy[2], lr_sensitivity[2], lr_specificity[2], lr_f1_scores[2])
)

# Plotting bar plots for comparison
par(mfrow = c(2, 2))  # Setting the layout for multiple plots

metrics_df_long <- reshape2::melt(metrics_df, id.vars = "Metric")

# Bar plot for Accuracy, Sensitivity, Specificity, and F1 Score
ggplot(metrics_df_long, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.7) +
  labs(title = "Metrics Comparison",
       x = "Metrics",
       y = "Values",
       fill = "Dataset") +
  scale_fill_manual(values = c("skyblue", "salmon")) +
  geom_text(aes(label = round(value, 3)), position = position_dodge(width = 0.9), vjust = -0.25) +
  theme_minimal()

library(gplots)
plot(as.table(lr_confusion_matrixxx), main = "Confusion Matrix")
plot(as.table(lr_confusion_matrix_val), main = "Confusion Matrix Validation")


# 3)DECISION TREE: Apply the decision tree on the data set WITHOUT normalization
independent_var <- heart_with_missing[, 1:13]
dependent_var <- heart_with_missing[, 14]

install.packages("rpart")
library(rpart)
decision_tree <- rpart(dependent_var ~. ,data = independent_var,method="class")
#print(decision_tree)
plot(decision_tree)
text(decision_tree)

dt_predictions <- predict(decision_tree, newdata = testDataa,type="class")
dt_pred_val <- predict(decision_tree, newdata = validationDataa, type="class")


#find the confusion matrix
dt_confusion_matrixxx <- table(dt_predictions,testDataa$target)
dt_confusion_matrixxx

dt_confusion_matrix_val <- table(dt_pred_val,validationDataa$target)
dt_confusion_matrix_val

#calculate the accuracy of original data set
dt_accuracy_originalll <- sum(diag(dt_confusion_matrixxx))/sum(dt_confusion_matrixxx)
cat("Accuracy for original data:",dt_accuracy_originalll)

dt_accuracy_original_val <- sum(diag(dt_confusion_matrix_val))/sum(dt_confusion_matrix_val)
cat("Accuracy for original data:",dt_accuracy_original_val)

#sensitivity
TP <- dt_confusion_matrixxx[2,2] #number of true positive values
FN <- dt_confusion_matrixxx[2,1] #number of false negative values

dt_sensitivity <- TP / (TP + FN) #sensitivity calculation
print(dt_sensitivity)


TP <- dt_confusion_matrix_val[2,2] #number of true positive values
FN <- dt_confusion_matrix_val[2,1] #number of false negative values

dt_sensitivity_val <- TP / (TP + FN) #sensitivity calculation
print(dt_sensitivity_val)

#specificity
TN <- dt_confusion_matrixxx[1,1] #number of true negative values 
FP <- dt_confusion_matrixxx[1,2] #number of false positive values

dt_specificity <- TN / (TN + FP) #specificity calculation
print(dt_specificity)


TN <- dt_confusion_matrix_val[1,1] #number of true negative values 
FP <- dt_confusion_matrix_val[1,2] #number of false positive values

dt_specificity_val <- TN / (TN + FP) #specificity calculation
print(dt_specificity_val)

#F1-Score
TP <- dt_confusion_matrixxx[2,2] #number of true positive values
FP <- dt_confusion_matrixxx[1,2] #number of false positive values
FN <- dt_confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

dt_f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(dt_f1_score)


TP <- dt_confusion_matrix_val[2,2] #number of true positive values
FP <- dt_confusion_matrix_val[1,2] #number of false positive values
FN <- dt_confusion_matrix_val[2,1] #number of false negative values
precision_val <- TP / (TP + FP)
recall_val <- TP / (TP + FN)

dt_f1_score_val <- (2 * (precision_val + recall_val)) / (precision_val + recall_val) #f1_score calculation
print(dt_f1_score_val)

#results of Decision Tree classification visualization:
# Storing metric values in variables as vectors
dt_accuracy <- c(dt_accuracy_originalll, dt_accuracy_original_val)
dt_sensitivity <- c(dt_sensitivity, dt_sensitivity_val)
dt_specificity <- c(dt_specificity, dt_specificity_val)
dt_f1_scores <- c(dt_f1_score, dt_f1_score_val)

# Creating a dataframe
metrics_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Test = c(dt_accuracy[1], dt_sensitivity[1], dt_specificity[1], dt_f1_scores[1]),
  Validation = c(dt_accuracy[2], dt_sensitivity[2], dt_specificity[2], dt_f1_scores[2])
)

# Plotting bar plots for comparison
par(mfrow = c(2, 2))  # Setting the layout for multiple plots

metrics_df_long <- reshape2::melt(metrics_df, id.vars = "Metric")

# Bar plot for Accuracy, Sensitivity, Specificity, and F1 Score
ggplot(metrics_df_long, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.7) +
  labs(title = "Metrics Comparison",
       x = "Metrics",
       y = "Values",
       fill = "Dataset") +
  scale_fill_manual(values = c("skyblue", "salmon")) +
  geom_text(aes(label = round(value, 3)), position = position_dodge(width = 0.9), vjust = -0.25) +
  theme_minimal()

library(gplots)
plot(as.table(dt_confusion_matrixxx), main = "Decision Tree Confusion Matrix")
plot(as.table(dt_confusion_matrix_val), main = "Decision Tree Confusion Matrix")


# 4)SVM: Apply the support vector machine on the data set WITHOUT normalization
library(e1071)
set.seed(123)
svm_model <- svm(target ~., data = trainDataa)

svm_pred <- predict(svm_model,testDataa)
svm_pred_val <- predict(svm_model,validationDataa)


#confusion matrix
svm_confusion_matrixxx <- table(svm_pred,testDataa$target)
print("Confusion matrix:")
print(svm_confusion_matrixxx)

svm_confusion_matrix_val <- table(svm_pred_val,validationDataa$target)
print("Confusion matrix val:")
print(svm_confusion_matrix_val)


#calculate the accuracy of original data set
svm_accuracy_originalll <- sum(diag(svm_confusion_matrixxx))/sum(svm_confusion_matrixxx)
cat("Accuracy for original data:",svm_accuracy_originalll)

svm_accuracy_original_val <- sum(diag(svm_confusion_matrix_val))/sum(svm_confusion_matrix_val)
cat("Accuracy for original data:",svm_accuracy_original_val)

#sensitivity
TP <- svm_confusion_matrixxx[2,2] #number of true positive values
FN <- svm_confusion_matrixxx[2,1] #number of false negative values

svm_sensitivity <- TP / (TP + FN) #sensitivity calculation
print(svm_sensitivity)

TP <- svm_confusion_matrix_val[2,2] #number of true positive values
FN <- svm_confusion_matrix_val[2,1] #number of false negative values

svm_sensitivity_val <- TP / (TP + FN) #sensitivity calculation
print(svm_sensitivity_val)

#specificity
TN <- svm_confusion_matrixxx[1,1] #number of true negative values 
FP <- svm_confusion_matrixxx[1,2] #number of false positive values

svm_specificity <- TN / (TN + FP) #specificity calculation
print(svm_specificity)

TN <- svm_confusion_matrix_val[1,1] #number of true negative values 
FP <- svm_confusion_matrix_val[1,2] #number of false positive values

svm_specificity_val <- TN / (TN + FP) #specificity calculation
print(svm_specificity_val)

#F1-Score
TP <- svm_confusion_matrixxx[2,2] #number of true positive values
FP <- svm_confusion_matrixxx[1,2] #number of false positive values
FN <- svm_confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

svm_f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(svm_f1_score)

TP <- svm_confusion_matrix_val[2,2] #number of true positive values
FP <- svm_confusion_matrix_val[1,2] #number of false positive values
FN <- svm_confusion_matrix_val[2,1] #number of false negative values
precision_val <- TP / (TP + FP)
recall_val <- TP / (TP + FN)

svm_f1_score_val <- (2 * (precision_val + recall_val)) / (precision_val + recall_val) #f1_score calculation
print(svm_f1_score_val)

# 
# #results of SVM classification visualization:
# # Creating a dataframe for visualization
# Store evaluation metrics in variables
accuracy_svm <- c(svm_accuracy_originalll, svm_accuracy_original_val)
sensitivity_svm <- c(svm_sensitivity, svm_sensitivity_val)
specificity_svm <- c(svm_specificity, svm_specificity_val)
f1_scores_svm <- c(svm_f1_score, svm_f1_score_val)

# Create a dataframe for plotting
metrics_svm_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Test = accuracy_svm,
  Validation = c(sensitivity_svm, specificity_svm, f1_scores_svm)
)

#Reshape the data for visualization
metrics_svm_df_long <- reshape2::melt(metrics_svm_df, id.vars = "Metric")

# Create a bar plot
ggplot(metrics_svm_df_long, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.7) +
  labs(title = "Metrics Comparison for SVM",
       x = "Metrics",
       y = "Values",
       fill = "Dataset") +
  scale_fill_manual(values = c("skyblue", "salmon")) +
  geom_text(aes(label = round(value, 3)), position = position_dodge(width = 0.9), vjust = -0.25) +
  theme_minimal()

library(gplots)
plot(as.table(svm_confusion_matrixxx), main = "SVM Confusion Matrix")
plot(as.table(svm_confusion_matrix_val), main = "SVM Validation Confusion Matrix")


# 5) GAUSSIAN NAIVE BAYES: Apply the gaussian naive bayes on the data set WITHOUT normalization
gnb_model <- naiveBayes(target ~., data = trainDataa)

gnb_pred <- predict(gnb_model, testDataa)
gnb_pred_val <- predict(gnb_model, validationDataa)


#find the confusion matrix
gnb_confusion_matrixxx <- table(gnb_pred,testDataa$target)
gnb_confusion_matrixxx

gnb_confusion_matrix_val <- table(gnb_pred_val,validationDataa$target)
gnb_confusion_matrix_val

#calculate the accuracy of original data set
gnb_accuracy_originalll <- sum(diag(gnb_confusion_matrixxx))/sum(gnb_confusion_matrixxx)
cat("Accuracy for original data:",gnb_accuracy_originalll)

gnb_accuracy_original_val <- sum(diag(gnb_confusion_matrix_val))/sum(gnb_confusion_matrix_val)
cat("Accuracy for original data:",gnb_accuracy_original_val)

#sensitivity
TP <- gnb_confusion_matrixxx[2,2] #number of true positive values
FN <- gnb_confusion_matrixxx[2,1] #number of false negative values

gnb_sensitivity <- TP / (TP + FN) #sensitivity calculation
print(gnb_sensitivity)

TP <- gnb_confusion_matrix_val[2,2] #number of true positive values
FN <- gnb_confusion_matrix_val[2,1] #number of false negative values

gnb_sensitivity_val <- TP / (TP + FN) #sensitivity calculation
print(gnb_sensitivity_val)

#specificity
TN <- gnb_confusion_matrixxx[1,1] #number of true negative values 
FP <- gnb_confusion_matrixxx[1,2] #number of false positive values

gnb_specificity <- TN / (TN + FP) #specificity calculation
print(gnb_specificity)


TN <- gnb_confusion_matrix_val[1,1] #number of true negative values 
FP <- gnb_confusion_matrix_val[1,2] #number of false positive values

gnb_specificity_val <- TN / (TN + FP) #specificity calculation
print(gnb_specificity_val)

#F1-Score
TP <- gnb_confusion_matrixxx[2,2] #number of true positive values
FP <- gnb_confusion_matrixxx[1,2] #number of false positive values
FN <- gnb_confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

gnb_f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(gnb_f1_score)

TP <- gnb_confusion_matrix_val[2,2] #number of true positive values
FP <- gnb_confusion_matrix_val[1,2] #number of false positive values
FN <- gnb_confusion_matrix_val[2,1] #number of false negative values
precision_val <- TP / (TP + FP)
recall_val <- TP / (TP + FN)

gnb_f1_score_val <- (2 * (precision_val + recall_val)) / (precision_val + recall_val) #f1_score calculation
print(gnb_f1_score_val)

#results of Gaussian Naive Bayes classification visualization:
#create the data frame which includes the metrics 
metrics_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Test = c(accuracy_originalll, sensitivity, specificity, f1_score),
  Validation = c(accuracy_original_val, sensitivity_val, specificity_val, f1_score_val)
)

# reshape for the long format
library(reshape2)
metrics_df_long <- melt(metrics_df, id.vars = "Metric")

# bar plot
library(ggplot2)
ggplot(metrics_df_long, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.7) +
  labs(title = "Metrics Comparison for Gaussian Naive Bayes",
       x = "Metrics",
       y = "Values",
       fill = "Dataset") +
  scale_fill_manual(values = c("skyblue", "salmon"))+
  geom_text(aes(label = round(value, 3)), position = position_dodge(width = 0.9), vjust = -0.25) +
  theme_minimal()

library(gplots)
plot(as.table(gnb_confusion_matrixxx), main = "Confusion Matrix")
plot(as.table(gnb_confusion_matrix_val), main = "Confusion Matrix Validation")


# 6) RANDOM FOREST: Apply the random forest classification method on the data set WITHOUT normalization
library(randomForest)
rf_model <- randomForest(target ~., data = trainDataa)

rf_pred <- predict(rf_model, newdata = testDataa)
rf_pred_val <- predict(rf_model, newdata = validationDataa)

#find the confusion matrix
rf_confusion_matrixxx <- table(rf_pred,testDataa$target)
rf_confusion_matrixxx

rf_confusion_matrix_val <- table(rf_pred_val,validationDataa$target)
rf_confusion_matrix_val

#calculate the accuracy of original data set
rf_accuracy_originalll <- sum(diag(rf_confusion_matrixxx))/sum(rf_confusion_matrixxx)
cat("Accuracy for original data:",rf_accuracy_originalll)

rf_accuracy_original_val <- sum(diag(rf_confusion_matrix_val))/sum(rf_confusion_matrix_val)
cat("Accuracy for original data:",rf_accuracy_original_val)

#sensitivity
TP <- rf_confusion_matrixxx[2,2] #number of true positive values
FN <- rf_confusion_matrixxx[2,1] #number of false negative values

rf_sensitivity <- TP / (TP + FN) #sensitivity calculation
print(rf_sensitivity)


TP <- rf_confusion_matrix_val[2,2] #number of true positive values
FN <- rf_confusion_matrix_val[2,1] #number of false negative values

rf_sensitivity_val <- TP / (TP + FN) #sensitivity calculation
print(rf_sensitivity_val)

#specificity
TN <- rf_confusion_matrixxx[1,1] #number of true negative values 
FP <- rf_confusion_matrixxx[1,2] #number of false positive values

rf_specificity <- TN / (TN + FP) #specificity calculation
print(rf_specificity)


TN <- rf_confusion_matrix_val[1,1] #number of true negative values 
FP <- rf_confusion_matrix_val[1,2] #number of false positive values

rf_specificity_val <- TN / (TN + FP) #specificity calculation
print(rf_specificity_val)

#F1-Score
TP <- rf_confusion_matrixxx[2,2] #number of true positive values
FP <- rf_confusion_matrixxx[1,2] #number of false positive values
FN <- rf_confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

rf_f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(rf_f1_score)

TP <- rf_confusion_matrix_val[2,2] #number of true positive values
FP <- rf_confusion_matrix_val[1,2] #number of false positive values
FN <- rf_confusion_matrix_val[2,1] #number of false negative values
precision_val <- TP / (TP + FP)
recall_val <- TP / (TP + FN)

rf_f1_score_val <- (2 * (precision_val + recall_val)) / (precision_val + recall_val) #f1_score calculation
print(rf_f1_score_val)

# create a data frame which includes the metrics
metrics_rf <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Test = c(accuracy_originalll, sensitivity, specificity, f1_score),
  Validation = c(accuracy_original_val, sensitivity_val, specificity_val, f1_score_val)
)

# reshape to long format
library(reshape2)
metrics_rf_long <- melt(metrics_rf, id.vars = "Metric")

# bar graph
library(ggplot2)
ggplot(metrics_rf_long, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.7) +
  labs(title = "Metrics Comparison for Random Forest",
       x = "Metrics",
       y = "Values",
       fill = "Dataset") +
  scale_fill_manual(values = c("skyblue", "salmon")) +
  geom_text(aes(label = round(value, 3)), position = position_dodge(width = 0.9), vjust = -0.25) +
  theme_minimal()

library(gplots)
plot(as.table(rf_confusion_matrixxx), main = "Confusion Matrix")
plot(as.table(rf_confusion_matrix_val), main = "Confusion Matrix Validation")
#end of the classification part

#-----------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------
#NORMALIZATION
names(heart_with_missing)
normalized_data <- scale(heart_with_missing[, c("age","sex","cp","trestbps","chol", "fbs", "restecg", "thalach","exang","oldpeak","slope", "ca","thal")])
#is.data.frame(normalized_data)

normalized_df <- data.frame(normalized_data,
                            target = heart_with_missing$target)
normalized_df
#summary(normalized_df) # compare the summary() results of these two data frames. Through this way we can come through that normalization has been apllied to the data frame
#summary(heart_with_missing)

#-----------------------------------------------------------------------------------------------
#After the normalization part, apply the all steps for the normalized data frame too.

#-----------------------------------------------------------------------------------------------
#VISUALISATION PART OF NORMALIZED DATA
library(ggplot2)
#***
M <- cor(normalized_df[, -13])  # Create correlation matrix
corrplot(M, method = "color")  #Draw a correlation chart

#***
png(file = "normalizedColumnsRelation.png") #it saves the .png format of the graph
pairs(normalized_df) #it shows the relationships between all columns as a matrix
dev.off()

#***
for (i in names(normalized_df))  #bar graph
{
  if (is.numeric(normalized_df[[i]]) && i!="target") {  #If the column is numeric and not the target column
    p <- ggplot(data = normalized_df, aes_string(x = i, y = "target")) +
      geom_bar(stat = "identity") +
      ggtitle(paste("Bar Plot of", i))
    print(p)
    Sys.sleep(2) #wait 2 seconds
  }
}

#***
for (col in names(normalized_df)) # density graph
{
  if (is.numeric(normalized_df[[col]]) && col != "target") {  # Eğer sütun numerik ise ve target sütunu değilse
    p <- ggplot(data = normalized_df, aes_string(x = col)) +
      geom_density() +
      ggtitle(paste("Density Plot of", col))
    print(p)
    Sys.sleep(2) #wait 2 seconds
  }
}
#end of the visualization part of the normalized data


#-----------------------------------------------------------------------------------------------
#CLASSIFICATION ON THE NORMALIZED DATAFRAME(6 types)
# 1) KNN: Apply the KNN algorithm on the data set WITH normalization 
#split the data set into train(70%),test(15%), and validation(15%)
set.seed(123)
trainIndex <- createDataPartition(data$target, p = 0.7, list = FALSE)
trainDataa <- data[trainIndex,]
temporaryData <- data[-trainIndex,]

validationIndex <- createDataPartition(temporaryData$target, p = 0.5, list = FALSE)
validationDataa <- temporaryData[validationIndex,]
norm_testDataa <- temporaryData[-validationIndex,] #real values

#check the number of rows of the splitted data
nrow(trainDataa) 
nrow(norm_testDataa)
nrow(validationDataa)

library(caret)
library(lattice)
library(class)

#apply the knn algorithm
norm_knn_modell <- knn(train = trainDataa[,-13], 
                  test = norm_testDataa[,-13],
                  cl = trainDataa$target,
                  k = 7)
knn_modell_val <- knn(train = trainDataa[,-13], 
                      test = validationDataa[,-13],
                      cl = trainDataa$target,
                      k = 7)

norm_knn_modell #predicted test values
knn_modell_val #predicted validation values
print(c(norm_testDataa$target)) #real values

#find the confusion matrix
norm_knn_confusion_matrixxx <- table(Reference = norm_testDataa$target,  
                                Prediction = norm_knn_modell)
cat("Confusion matrix of test data:")
norm_knn_confusion_matrixxx

norm_knn_confusion_matrix_val <- table(Reference = norm_testDataa$target,  
                                  Prediction = knn_modell_val)
cat("Confusion matrix of test data:")
norm_knn_confusion_matrix_val

#calculate the accuracy of original data set
norm_knn_accuracy_originalll <- sum(diag(norm_knn_confusion_matrixxx))/sum(norm_knn_confusion_matrixxx)
cat("Accuracy for original data:",norm_knn_accuracy_originalll)

norm_knn_accuracy_original_val <- sum(diag(norm_knn_confusion_matrix_val))/sum(norm_knn_confusion_matrix_val)
cat("Accuracy for original data:",norm_knn_accuracy_original_val)

#sensitivity
TP <- norm_knn_confusion_matrixxx[2,2] #number of true positive values
FN <- norm_knn_confusion_matrixxx[2,1] #number of false negative values

norm_knn_sensitivity <- TP / (TP + FN) #sensitivity calculation
print(norm_knn_sensitivity)

TP <- norm_knn_confusion_matrix_val[2,2] #number of true positive values
FN <- norm_knn_confusion_matrix_val[2,1] #number of false negative values

norm_knn_sensitivity_val <- TP / (TP + FN) #sensitivity calculation
print(norm_knn_sensitivity_val)

#specificity
TN <- norm_knn_confusion_matrixxx[1,1] #number of true negative values 
FP <- norm_knn_confusion_matrixxx[1,2] #number of false positive values

norm_knn_specificity <- TN / (TN + FP) #specificity calculation
print(norm_knn_specificity)

TN <- norm_knn_confusion_matrix_val[1,1] #number of true negative values 
FP <- norm_knn_confusion_matrix_val[1,2] #number of false positive values

norm_knn_specificity_val <- TN / (TN + FP) #specificity calculation
print(norm_knn_specificity_val)

#F1-Score
TP <- norm_knn_confusion_matrixxx[2,2] #number of true positive values
FP <- norm_knn_confusion_matrixxx[1,2] #number of false positive values
FN <- norm_knn_confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

norm_knn_f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(norm_knn_f1_score)

TP <- norm_knn_confusion_matrix_val[2,2] #number of true positive values
FP <- norm_knn_confusion_matrix_val[1,2] #number of false positive values
FN <- norm_knn_confusion_matrix_val[2,1] #number of false negative values
precision_val <- TP / (TP + FP)
recall_val <- TP / (TP + FN)

norm_knn_f1_score_val <- (2 * (precision_val + recall_val)) / (precision_val + recall_val) #f1_score calculation
print(norm_knn_f1_score_val)

# #results of KNN classification visualization:
# Storing metric values in variables as vectors
norm_knn_accuracy <- c(norm_knn_accuracy_originalll, norm_knn_accuracy_original_val)
norm_knn_sensitivity <- c(norm_knn_sensitivity, norm_knn_sensitivity_val)
norm_knn_specificity <- c(norm_knn_specificity, norm_knn_specificity_val)
norm_knn_f1_scores <- c(norm_knn_f1_score, norm_knn_f1_score_val)

# Creating a dataframe
metrics_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Test = c(norm_knn_accuracy[1], norm_knn_sensitivity[1], norm_knn_specificity[1], norm_knn_f1_scores[1]),
  Validation = c(norm_knn_accuracy[2], norm_knn_sensitivity[2], norm_knn_specificity[2], norm_knn_f1_scores[2])
)

# Plotting bar plots for comparison
par(mfrow = c(2, 2))  # Setting the layout for multiple plots

metrics_df_long <- reshape2::melt(metrics_df, id.vars = "Metric")

# Bar plot for Accuracy, Sensitivity, Specificity, and F1 Score
ggplot(metrics_df_long, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.7) +
  labs(title = "Metrics Comparison",
       x = "Metrics",
       y = "Values",
       fill = "Dataset") +
  scale_fill_manual(values = c("skyblue", "salmon")) +
  geom_text(aes(label = round(value, 3)), position = position_dodge(width = 0.9), vjust = -0.25) +
  theme_minimal()

library(gplots)
plot(as.table(norm_knn_confusion_matrixxx), main = "Confusion Matrix")
plot(as.table(norm_knn_confusion_matrix_val), main = "Confusion Matrix Validation")


# 2)LOGISTIC REGRESSION: Apply the logistic regression on the data set WITH normalization
independent_var <- heart_with_missing[, 1:13]
dependent_var <- heart_with_missing[, 14]

logit_model <- glm(dependent_var ~ ., data = heart_with_missing, family = binomial) #compose the logistic regression model
logit_model

norm_lr_predictions <- predict(logit_model, newdata = heart_with_missing, type="response")
norm_lr_predicted_classes <- ifelse(predictions > 0.5,1,0)

norm_lr_pred_val <- predict(logit_model, newdata = validationDataa, type="response")
norm_lr_predicted_classes <- ifelse(norm_lr_predictions > 0.5,1,0)

#find the confusion matrix
norm_lr_confusion_matrixxx <- table(norm_lr_predicted_classes,heart_with_missing$target)
print(norm_lr_confusion_matrixxx)

norm_lr_confusion_matrix_val <- table(norm_lr_pred_val,validationDataa$target)
print(norm_lr_confusion_matrix_val)

#calculate the accuracy of original data set
norm_lr_accuracy_originalll <- norm_lr_confusion_matrixxx['Accuracy']
print(paste("Accuracy:",norm_lr_accuracy_originalll))

norm_lr_accuracy_original_val <- norm_lr_confusion_matrix_val['Accuracy validation']
print(paste("Accuracy:",norm_lr_accuracy_original_val))

#sensitivity
norm_lr_sensitivity <- norm_lr_confusion_matrixxx['Sensitivity']
print(paste("Sensitivity",norm_lr_sensitivity))

norm_lr_sensitivity_val <- norm_lr_confusion_matrix_val['Sensitivity validation']
print(paste("Sensitivity",norm_lr_sensitivity_val))

#specificity
norm_lr_specificity <- norm_lr_confusion_matrixxx['Specificity']
print(paste("Specificity",norm_lr_specificity))

norm_lr_specificity_val <- norm_lr_confusion_matrix_val['Specificity']
print(paste("Specificity",norm_lr_specificity_val))

#F1-Score
norm_lr_f1_score <- norm_lr_confusion_matrixxx['F1']
print(paste("F1 Score",norm_lr_f1_score))

norm_lr_f1_score_val <- norm_lr_confusion_matrix_val['F1']
print(paste("F1 Score",norm_lr_f1_score_val))

#results of Logistic Regression classification visualization:
# Storing metric values in variables as vectors
norm_lr_accuracy <- c(norm_lr_accuracy_originalll, norm_lr_accuracy_original_val)
norm_lr_sensitivity <- c(norm_lr_sensitivity,norm_lr_sensitivity_val)
norm_lr_specificity <- c(norm_lr_specificity, norm_lr_specificity_val)
norm_lr_f1_scores <- c(norm_lr_f1_score, norm_lr_f1_score_val)

# Creating a dataframe
metrics_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Test = c(norm_lr_accuracy[1], norm_lr_sensitivity[1], norm_lr_specificity[1], norm_lr_f1_scores[1]),
  Validation = c(norm_lr_accuracy[2], norm_lr_sensitivity[2], norm_lr_specificity[2], norm_lr_f1_scores[2])
)

# Plotting bar plots for comparison
par(mfrow = c(2, 2))  # Setting the layout for multiple plots

metrics_df_long <- reshape2::melt(metrics_df, id.vars = "Metric")

# Bar plot for Accuracy, Sensitivity, Specificity, and F1 Score
ggplot(metrics_df_long, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.7) +
  labs(title = "Metrics Comparison",
       x = "Metrics",
       y = "Values",
       fill = "Dataset") +
  scale_fill_manual(values = c("skyblue", "salmon")) +
  geom_text(aes(label = round(value, 3)), position = position_dodge(width = 0.9), vjust = -0.25) +
  theme_minimal()

library(gplots)
plot(as.table(norm_lr_confusion_matrixxx), main = "Confusion Matrix")
plot(as.table(norm_lr_confusion_matrix_val), main = "Confusion Matrix Validation")


# 3)DECISION TREE: Apply the decision tree on the data set WITH normalization
independent_var <- heart_with_missing[, 1:13]
dependent_var <- heart_with_missing[, 14]

install.packages("rpart")
library(rpart)
norm_decision_tree <- rpart(dependent_var ~. ,data = independent_var,method="class")
#print(decision_tree)
plot(decision_tree)
text(decision_tree)

norm_dt_predictions <- predict(norm_decision_tree, newdata = testDataa,type="class")
norm_dt_pred_val <- predict(norm_decision_tree, newdata = validationDataa, type="class")


#find the confusion matrix
norm_dt_confusion_matrixxx <- table(norm_dt_predictions,testDataa$target)
norm_dt_confusion_matrixxx

norm_dt_confusion_matrix_val <- table(norm_dt_pred_val,validationDataa$target)
norm_dt_confusion_matrix_val

#calculate the accuracy of original data set
norm_dt_accuracy_originalll <- sum(diag(norm_dt_confusion_matrixxx))/sum(norm_dt_confusion_matrixxx)
cat("Accuracy for original data:",norm_dt_accuracy_originalll)

norm_dt_accuracy_original_val <- sum(diag(norm_dt_confusion_matrix_val))/sum(norm_dt_confusion_matrix_val)
cat("Accuracy for original data:",norm_dt_accuracy_original_val)

#sensitivity
TP <- norm_dt_confusion_matrixxx[2,2] #number of true positive values
FN <- norm_dt_confusion_matrixxx[2,1] #number of false negative values

norm_dt_sensitivity <- TP / (TP + FN) #sensitivity calculation
print(norm_dt_sensitivity)


TP <- norm_dt_confusion_matrix_val[2,2] #number of true positive values
FN <- norm_dt_confusion_matrix_val[2,1] #number of false negative values

norm_dt_sensitivity_val <- TP / (TP + FN) #sensitivity calculation
print(norm_dt_sensitivity_val)

#specificity
TN <- norm_dt_confusion_matrixxx[1,1] #number of true negative values 
FP <- norm_dt_confusion_matrixxx[1,2] #number of false positive values

norm_dt_specificity <- TN / (TN + FP) #specificity calculation
print(norm_dt_specificity)


TN <- norm_dt_confusion_matrix_val[1,1] #number of true negative values 
FP <- norm_dt_confusion_matrix_val[1,2] #number of false positive values

norm_dt_specificity_val <- TN / (TN + FP) #specificity calculation
print(norm_dt_specificity_val)

#F1-Score
TP <- norm_dt_confusion_matrixxx[2,2] #number of true positive values
FP <- norm_dt_confusion_matrixxx[1,2] #number of false positive values
FN <- norm_dt_confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

norm_dt_f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(norm_dt_f1_score)


TP <- norm_dt_confusion_matrix_val[2,2] #number of true positive values
FP <- norm_dt_confusion_matrix_val[1,2] #number of false positive values
FN <- norm_dt_confusion_matrix_val[2,1] #number of false negative values
precision_val <- TP / (TP + FP)
recall_val <- TP / (TP + FN)

norm_dt_f1_score_val <- (2 * (precision_val + recall_val)) / (precision_val + recall_val) #f1_score calculation
print(norm_dt_f1_score_val)

#results of Decision Tree classification visualization:
# Storing metric values in variables as vectors
norm_dt_accuracy <- c(norm_dt_accuracy_originalll, norm_dt_accuracy_original_val)
norm_dt_sensitivity <- c(norm_dt_sensitivity, norm_dt_sensitivity_val)
norm_dt_specificity <- c(norm_dt_specificity, norm_dt_specificity_val)
norm_dt_f1_scores <- c(norm_dt_f1_score, norm_dt_f1_score_val)

# Creating a dataframe
metrics_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Test = c(norm_dt_accuracy[1], norm_dt_sensitivity[1], norm_dt_specificity[1], norm_dt_f1_scores[1]),
  Validation = c(norm_dt_accuracy[2], norm_dt_sensitivity[2], norm_dt_specificity[2], norm_dt_f1_scores[2])
)

# Plotting bar plots for comparison
par(mfrow = c(2, 2))  # Setting the layout for multiple plots

metrics_df_long <- reshape2::melt(metrics_df, id.vars = "Metric")

# Bar plot for Accuracy, Sensitivity, Specificity, and F1 Score
ggplot(metrics_df_long, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.7) +
  labs(title = "Metrics Comparison",
       x = "Metrics",
       y = "Values",
       fill = "Dataset") +
  scale_fill_manual(values = c("skyblue", "salmon")) +
  geom_text(aes(label = round(value, 3)), position = position_dodge(width = 0.9), vjust = -0.25) +
  theme_minimal()

library(gplots)
plot(as.table(norm_dt_confusion_matrixxx), main = "Decision Tree Confusion Matrix")
plot(as.table(norm_dt_confusion_matrix_val), main = "Decision Tree Val Confusion Matrix")


# 4)SVM: Apply the support vector machine on the data set WITH normalization
library(e1071)
set.seed(123)
svm_model <- svm(target ~., data = trainDataa)

norm_svm_pred <- predict(svm_model,norm_testDataa)
norm_svm_pred_val <- predict(svm_model,validationDataa)


#confusion matrix
norm_svm_confusion_matrixxx <- table(norm_svm_pred,testDataa$target)
print("Confusion matrix:")
print(norm_svm_confusion_matrixxx)

norm_svm_confusion_matrix_val <- table(norm_svm_pred_val,validationDataa$target)
print("Confusion matrix val:")
print(norm_svm_confusion_matrix_val)


#calculate the accuracy of original data set
norm_svm_accuracy_originalll <- sum(diag(norm_svm_confusion_matrixxx))/sum(norm_svm_confusion_matrixxx)
cat("Accuracy for original data:",norm_svm_accuracy_originalll)

norm_svm_accuracy_original_val <- sum(diag(norm_svm_confusion_matrix_val))/sum(norm_svm_confusion_matrix_val)
cat("Accuracy for original data:",norm_svm_accuracy_original_val)

#sensitivity
TP <- norm_svm_confusion_matrixxx[2,2] #number of true positive values
FN <- norm_svm_confusion_matrixxx[2,1] #number of false negative values

norm_svm_sensitivity <- TP / (TP + FN) #sensitivity calculation
print(norm_svm_sensitivity)

TP <- norm_svm_confusion_matrix_val[2,2] #number of true positive values
FN <- norm_svm_confusion_matrix_val[2,1] #number of false negative values

norm_svm_sensitivity_val <- TP / (TP + FN) #sensitivity calculation
print(norm_svm_sensitivity_val)

#specificity
TN <- norm_svm_confusion_matrixxx[1,1] #number of true negative values 
FP <- norm_svm_confusion_matrixxx[1,2] #number of false positive values

norm_svm_specificity <- TN / (TN + FP) #specificity calculation
print(norm_svm_specificity)

TN <- norm_svm_confusion_matrix_val[1,1] #number of true negative values 
FP <- norm_svm_confusion_matrix_val[1,2] #number of false positive values

norm_svm_specificity_val <- TN / (TN + FP) #specificity calculation
print(norm_svm_specificity_val)

#F1-Score
TP <- norm_svm_confusion_matrixxx[2,2] #number of true positive values
FP <- norm_svm_confusion_matrixxx[1,2] #number of false positive values
FN <- norm_svm_confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

norm_svm_f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(norm_svm_f1_score)

TP <- norm_svm_confusion_matrix_val[2,2] #number of true positive values
FP <- norm_svm_confusion_matrix_val[1,2] #number of false positive values
FN <- norm_svm_confusion_matrix_val[2,1] #number of false negative values
precision_val <- TP / (TP + FP)
recall_val <- TP / (TP + FN)

norm_svm_f1_score_val <- (2 * (precision_val + recall_val)) / (precision_val + recall_val) #f1_score calculation
print(norm_svm_f1_score_val)

#results of SVM classification visualization:
# Creating a dataframe for visualization
metrics_df <- data.frame(
  Metric = c("Accuracy","Sensitivity", "Specificity", "F1 Score"),
  Value = c(accuracy,sensitivity, specificity, f1_score)
)

# Creating a bar plot
library(ggplot2)

ggplot(metrics_df, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.5, color = "black") +
  geom_text(aes(label = round(Value, digits = 3)), vjust = -0.5, size = 4) +
  labs(title = "SVM Visualization",
       x = "Metric",
       y = "Value") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  )

library(gplots)
plot(as.table(norm_svm_confusion_matrixxx), main = "SVM Confusion Matrix")


# Store evaluation metrics in variables
norm_accuracy_svm <- c(norm_svm_accuracy_originalll, norm_svm_accuracy_original_val)
norm_sensitivity_svm <- c(norm_svm_sensitivity, norm_svm_sensitivity_val)
norm_specificity_svm <- c(norm_svm_specificity, norm_svm_specificity_val)
norm_f1_scores_svm <- c(norm_svm_f1_score, norm_svm_f1_score_val)

# Create a dataframe for plotting
metrics_svm_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Test = accuracy_svm,
  Validation = c(norm_sensitivity_svm, norm_specificity_svm, norm_f1_scores_svm)
)

#Reshape the data for visualization
metrics_svm_df_long <- reshape2::melt(metrics_svm_df, id.vars = "Metric")

# Create a bar plot
ggplot(metrics_svm_df_long, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.7) +
  labs(title = "Metrics Comparison for SVM",
       x = "Metrics",
       y = "Values",
       fill = "Dataset") +
  scale_fill_manual(values = c("skyblue", "salmon")) +
  geom_text(aes(label = round(value, 3)), position = position_dodge(width = 0.9), vjust = -0.25) +
  theme_minimal()

library(gplots)
plot(as.table(norm_svm_confusion_matrixxx), main = "Decision Tree Confusion Matrix")
plot(as.table(norm_svm_confusion_matrix_val), main = "Decision Tree Val Confusion Matrix")


# 5) GAUSSIAN NAIVE BAYES: Apply the gaussian naive bayes on the data set WITH normalization
norm_gnb_model <- naiveBayes(target ~., data = trainDataa)

norm_gnb_pred <- predict(norm_gnb_model, norm_testDataa)
norm_gnb_pred_val <- predict(norm_gnb_model, validationDataa)


#find the confusion matrix
norm_gnb_confusion_matrixxx <- table(norm_gnb_pred,norm_testDataa$target)
norm_gnb_confusion_matrixxx

norm_gnb_confusion_matrix_val <- table(norm_gnb_pred_val,validationDataa$target)
norm_gnb_confusion_matrix_val

#calculate the accuracy of original data set
norm_gnb_accuracy_originalll <- sum(diag(norm_gnb_confusion_matrixxx))/sum(norm_gnb_confusion_matrixxx)
cat("Accuracy for original data:",norm_gnb_accuracy_originalll)

norm_gnb_accuracy_original_val <- sum(diag(norm_gnb_confusion_matrix_val))/sum(norm_gnb_confusion_matrix_val)
cat("Accuracy for original data:",norm_gnb_accuracy_original_val)

#sensitivity
TP <- norm_gnb_confusion_matrixxx[2,2] #number of true positive values
FN <- norm_gnb_confusion_matrixxx[2,1] #number of false negative values

norm_gnb_sensitivity <- TP / (TP + FN) #sensitivity calculation
print(norm_gnb_sensitivity)

TP <- norm_gnb_confusion_matrix_val[2,2] #number of true positive values
FN <- norm_gnb_confusion_matrix_val[2,1] #number of false negative values

norm_gnb_sensitivity_val <- TP / (TP + FN) #sensitivity calculation
print(norm_gnb_sensitivity_val)

#specificity
TN <- norm_gnb_confusion_matrixxx[1,1] #number of true negative values 
FP <- norm_gnb_confusion_matrixxx[1,2] #number of false positive values

norm_gnb_specificity <- TN / (TN + FP) #specificity calculation
print(norm_gnb_specificity)


TN <- norm_gnb_confusion_matrix_val[1,1] #number of true negative values 
FP <- norm_gnb_confusion_matrix_val[1,2] #number of false positive values

norm_gnb_specificity_val <- TN / (TN + FP) #specificity calculation
print(norm_gnb_specificity_val)

#F1-Score
TP <- norm_gnb_confusion_matrixxx[2,2] #number of true positive values
FP <- norm_gnb_confusion_matrixxx[1,2] #number of false positive values
FN <- norm_gnb_confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

norm_gnb_f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(norm_gnb_f1_score)

TP <- norm_gnb_confusion_matrix_val[2,2] #number of true positive values
FP <- norm_gnb_confusion_matrix_val[1,2] #number of false positive values
FN <- norm_gnb_confusion_matrix_val[2,1] #number of false negative values
precision_val <- TP / (TP + FP)
recall_val <- TP / (TP + FN)

norm_gnb_f1_score_val <- (2 * (precision_val + recall_val)) / (precision_val + recall_val) #f1_score calculation
print(norm_gnb_f1_score_val)

#results of Gaussian Naive Bayes classification visualization:
#create the data frame which includes the metrics 
metrics_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Test = c(accuracy_originalll, sensitivity, specificity, f1_score),
  Validation = c(accuracy_original_val, sensitivity_val, specificity_val, f1_score_val)
)

# reshape for the long format
library(reshape2)
metrics_df_long <- melt(metrics_df, id.vars = "Metric")

# bar plot
library(ggplot2)
ggplot(metrics_df_long, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.7) +
  labs(title = "Metrics Comparison for Gaussian Naive Bayes",
       x = "Metrics",
       y = "Values",
       fill = "Dataset") +
  scale_fill_manual(values = c("skyblue", "salmon"))+
  geom_text(aes(label = round(value, 3)), position = position_dodge(width = 0.9), vjust = -0.25) +
  theme_minimal()

library(gplots)
plot(as.table(norm_gnb_confusion_matrixxx), main = "Confusion Matrix")
plot(as.table(norm_gnb_confusion_matrix_val), main = "Confusion Matrix Validation")


# 6) RANDOM FOREST: Apply the random forest classification method on the data set WITH normalization
library(randomForest)
rf_model <- randomForest(target ~., data = trainDataa)

norm_rf_pred <- predict(rf_model, newdata = norm_testDataa)
norm_rf_pred_val <- predict(rf_model, newdata = validationDataa)

#find the confusion matrix
norm_rf_confusion_matrixxx <- table(norm_rf_pred,norm_testDataa$target)
norm_rf_confusion_matrixxx

norm_rf_confusion_matrix_val <- table(norm_rf_pred_val,validationDataa$target)
norm_rf_confusion_matrix_val

#calculate the accuracy of original data set
norm_rf_accuracy_originalll <- sum(diag(norm_rf_confusion_matrixxx))/sum(norm_rf_confusion_matrixxx)
cat("Accuracy for original data:",norm_rf_accuracy_originalll)

norm_rf_accuracy_original_val <- sum(diag(norm_rf_confusion_matrix_val))/sum(norm_rf_confusion_matrix_val)
cat("Accuracy for original data:",norm_rf_accuracy_original_val)

#sensitivity
TP <- norm_rf_confusion_matrixxx[2,2] #number of true positive values
FN <- norm_rf_confusion_matrixxx[2,1] #number of false negative values

norm_rf_sensitivity <- TP / (TP + FN) #sensitivity calculation
print(norm_rf_sensitivity)


TP <- norm_rf_confusion_matrix_val[2,2] #number of true positive values
FN <- norm_rf_confusion_matrix_val[2,1] #number of false negative values

norm_rf_sensitivity_val <- TP / (TP + FN) #sensitivity calculation
print(norm_rf_sensitivity_val)

#specificity
TN <- norm_rf_confusion_matrixxx[1,1] #number of true negative values 
FP <- norm_rf_confusion_matrixxx[1,2] #number of false positive values

norm_rf_specificity <- TN / (TN + FP) #specificity calculation
print(norm_rf_specificity)


TN <- norm_rf_confusion_matrix_val[1,1] #number of true negative values 
FP <- norm_rf_confusion_matrix_val[1,2] #number of false positive values

norm_rf_specificity_val <- TN / (TN + FP) #specificity calculation
print(norm_rf_specificity_val)

#F1-Score
TP <- norm_rf_confusion_matrixxx[2,2] #number of true positive values
FP <- norm_rf_confusion_matrixxx[1,2] #number of false positive values
FN <- norm_rf_confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

norm_rf_f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(norm_rf_f1_score)

TP <- norm_rf_confusion_matrix_val[2,2] #number of true positive values
FP <- norm_rf_confusion_matrix_val[1,2] #number of false positive values
FN <- norm_rf_confusion_matrix_val[2,1] #number of false negative values
precision_val <- TP / (TP + FP)
recall_val <- TP / (TP + FN)

norm_rf_f1_score_val <- (2 * (precision_val + recall_val)) / (precision_val + recall_val) #f1_score calculation
print(norm_rf_f1_score_val)

# create a data frame which includes the metrics
metrics_rf <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Test = c(accuracy_originalll, sensitivity, specificity, f1_score),
  Validation = c(accuracy_original_val, sensitivity_val, specificity_val, f1_score_val)
)

# reshape to long format
library(reshape2)
metrics_rf_long <- melt(metrics_rf, id.vars = "Metric")

# bar graph
library(ggplot2)
ggplot(metrics_rf_long, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.7) +
  labs(title = "Metrics Comparison for Random Forest",
       x = "Metrics",
       y = "Values",
       fill = "Dataset") +
  scale_fill_manual(values = c("skyblue", "salmon")) +
  geom_text(aes(label = round(value, 3)), position = position_dodge(width = 0.9), vjust = -0.25) +
  theme_minimal()

library(gplots)
plot(as.table(norm_rf_confusion_matrixxx), main = "Confusion Matrix")
plot(as.table(rf_confusion_matrix_val), main = "Confusion Matrix Validation")


#end of the normalized data classification part


#-----------------------------------------------------------------------------------------------
#FEATURE EXTRACTION(4 types)

# 1)PCA(Principal Component Analysis): used to reduce variability in multidimensional data sets and reveal basic structures in the data set.
pca_result <- prcomp(normalized_df[,-14],scale=TRUE)
summary(pca_result)
plot(pca_result, type = "l")

# 2)LDA(Linear Discriminant Analysis): dimension reduction specialist used in appearance problems.
library(MASS)
lda_result <- lda(target ~ ., data = normalized_df)
plot(lda_result)

# 3)Recursive Feature elimination : used to determine the features that most affect the performance of a model.
library(caret)
library(ggplot2)
library(lattice)
control <- rfeControl(functions = rfFuncs, method = "cv", number=5)
results <- rfe(normalized_df[,-14],normalized_df$target,sizes = c(1:13), rfeControl = control)#lists the most affected classes
print(results)

selected_features <- predict(results, normalized_df[,-14]) 
print(selected_features) #outputs of the selected best features(most affected features)

# 4)Factor Analysis:used to describe observable relationships between variables and represent these relationships with a smaller number of factors
library(psych)
factor_analysis <- fa(normalized_df, nfactors =5 , rotate= "varimax")
print(factor_analysis)

#-----------------------------------------------------------------------------------------------
#BEST MODEL
# Function to calculate accuracy
calc_accuracy <- function(actual, predicted) {
  return(sum(actual == predicted) / length(actual))
}

# Calculate accuracy for each model on test data
model_accuracies <- c(
  KNN = calc_accuracy(norm_testDataa$target, norm_knn_modell),
  DecisionTree = calc_accuracy(norm_testDataa$target, norm_dt_predictions),
  #SVM = calc_accuracy(norm_testDataa$target, norm_svm_pred),
  LogisticRegression = calc_accuracy(norm_testDataa$target, norm_lr_predictions),
  GaussianNaiveBayes = calc_accuracy(norm_testDataa$target, norm_gnb_pred),
  RandomForest = calc_accuracy(norm_testDataa$target,norm_rf_pred)
)

# Find the model with the highest accuracy
best_model <- names(which.max(model_accuracies))
best_model



