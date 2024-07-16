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

#apply the knn algorithm
knn_modell <- knn(train = trainDataa[,1:13], 
                  test = testDataa[,1:13],
                  cl = trainDataa$target,
                  k = 7)

knn_modell #predicted values
print(c(testDataa$target)) #real values

#find the confusion matrix
confusion_matrixxx <- table(Reference = testDataa$target,  
                          Prediction = knn_modell)
cat("Confusion matrix of original data:")
confusion_matrixxx

#calculate the accuracy of original data set
accuracy_originalll <- sum(diag(confusion_matrixxx))/sum(confusion_matrixxx)
cat("Accuracy for original data:",accuracy_originalll)

#sensitivity
TP <- confusion_matrixxx[2,2] #number of true positive values
FN <- confusion_matrixxx[2,1] #number of false negative values

sensitivity <- TP / (TP + FN) #sensitivity calculation
print(sensitivity)

#specificity
TN <- confusion_matrixxx[1,1] #number of true negative values 
FP <- confusion_matrixxx[1,2] #number of false positive values

specificity <- TN / (TN + FP) #specificity calculation
print(specificity)

#F1-Score
TP <- confusion_matrixxx[2,2] #number of true positive values
FP <- confusion_matrixxx[1,2] #number of false positive values
FN <- confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(f1_score)

#results of KNN classification visualization:
sensitivity_values <- c(sensitivity)
specificity_values <- c(specificity) 
f1_score_values <- c(f1_score)  

labels <- c("Sensitivity", "Specificity", "f1-Score")

barplot_matrix <- matrix(c(sensitivity_values, specificity_values, f1_score_values), nrow = 3, byrow = TRUE)
barplot(t(barplot_matrix), beside = TRUE, col = c("orange", "purple", "pink"), 
        main = "Results of the KNN", xlab = "Metrics", ylab = "Values",
        names.arg = labels)

library(gplots)
plot(as.table(confusion_matrixxx), main = "Confusion Matrix")

# 2)LOGISTIC REGRESSION: Apply the logistic regression on the data set WITHOUT normalization
independent_var <- heart_with_missing[, 1:13]
dependent_var <- heart_with_missing[, 14]

logit_model <- glm(dependent_var ~ ., data = heart_with_missing, family = binomial) #compose the logistic regression model
logit_model

predictions <- predict(logit_model, newdata = heart_with_missing, type="response")
predicted_classes <- ifelse(predictions > 0.5,1,0)

#find the confusion matrix
confusion_matrixxx <- table(predicted_classes,dependent_var)
print(confusion_matrixxx)

#calculate the accuracy of original data set
accuracy_originalll <- confusion_matrixxx['Accuracy']
print(paste("Accuracy:",accuracy_originalll))

#sensitivity
sensitivity <- confusion_matrixxx['Sensitivity']
print(paste("Sensitivity",sensitivity))

#specificity
specificity <- confusion_matrixxx['Specificity']
print(paste("Specificity",specificity))

#F1-Score
f1_score <- confusion_matrixxx['F1']
print(paste("F1 Score",f1_score))

#results of Logistic Regression classification visualization:

library(ggplot2)
# Creating a dataframe for visualization
metrics_df <- data.frame(
  Metric = c("Accuracy","Sensitivity", "Specificity", "F1 Score"),
  Value = c(as.numeric(accuracy),as.numeric(sensitivity), as.numeric(specificity), as.numeric(f1_score))
)

# Creating a bar plot
ggplot(metrics_df, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.5, color = "black") +
  labs(title = "Logistic Regression Results",
       x = "Metric",
       y = "Value") +
  geom_text(aes(label = sprintf("%.3f", Value)), vjust = -0.5, size = 4) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  )

library(gplots)
plot(as.table(confusion_matrixxx), main = "Logistic Regression Confusion Matrix")


# 3)DECISION TREE: Apply the decision tree on the data set WITHOUT normalization
independent_var <- heart_with_missing[, 1:13]
dependent_var <- heart_with_missing[, 14]

install.packages("rpart")
library(rpart)
decision_tree <- rpart(dependent_var ~. ,data = independent_var,method="class")
#print(decision_tree)
plot(decision_tree)
text(decision_tree)

predictions <- predict(decision_tree, newdata = heart_with_missing,type="class")

#find the confusion matrix
confusion_matrixxx <- table(predictions,dependent_var)
confusion_matrixxx

#calculate the accuracy of original data set
accuracy_originalll <- sum(diag(confusion_matrixxx))/sum(confusion_matrixxx)
cat("Accuracy for original data:",accuracy_originalll)

#sensitivity
TP <- confusion_matrixxx[2,2] #number of true positive values
FN <- confusion_matrixxx[2,1] #number of false negative values

sensitivity <- TP / (TP + FN) #sensitivity calculation
print(sensitivity)

#specificity
TN <- confusion_matrixxx[1,1] #number of true negative values 
FP <- confusion_matrixxx[1,2] #number of false positive values

specificity <- TN / (TN + FP) #specificity calculation
print(specificity)

#F1-Score
TP <- confusion_matrixxx[2,2] #number of true positive values
FP <- confusion_matrixxx[1,2] #number of false positive values
FN <- confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(f1_score)

#results of Decision Tree classification visualization:
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
  labs(title = "Decision Tree Visualization",
       x = "Metric",
       y = "Value") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  )

library(gplots)
plot(as.table(confusion_matrixxx), main = "Decision Tree Confusion Matrix")


# 4)SVM: Apply the support vector machine on the data set WITHOUT normalization
library(e1071)
set.seed(123)
svm_model <- svm(target ~., data = heart_with_missing)

predictions <- predict(svm_model,heart_with_missing)

#confusion matrix
confusion_matrixxx <- table(predictions,heart_with_missing$target)
print("Confusion matrix:")
print(confusion_matrixxx)


#calculate the accuracy of original data set
accuracy_originalll <- sum(diag(confusion_matrixxx))/sum(confusion_matrixxx)
cat("Accuracy for original data:",accuracy_originalll)

#sensitivity
TP <- confusion_matrixxx[2,2] #number of true positive values
FN <- confusion_matrixxx[2,1] #number of false negative values

sensitivity <- TP / (TP + FN) #sensitivity calculation
print(sensitivity)

#specificity
TN <- confusion_matrixxx[1,1] #number of true negative values 
FP <- confusion_matrixxx[1,2] #number of false positive values

specificity <- TN / (TN + FP) #specificity calculation
print(specificity)

#F1-Score
TP <- confusion_matrixxx[2,2] #number of true positive values
FP <- confusion_matrixxx[1,2] #number of false positive values
FN <- confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(f1_score)


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
plot(as.table(confusion_matrixxx), main = "SVM Confusion Matrix")


# 5) GAUSSIAN NAIVE BAYES: Apply the gaussian naive bayes on the data set WITHOUT normalization
gnb_model <- naiveBayes(target ~., data = trainDataa)

#predictions <- predict(gnb_model, testDataa[, -which(names(testDataa)=="Class")])
predictions <- predict(gnb_model, newdata = heart_with_missing)

#find the confusion matrix
confusion_matrixxx <- table(predictions,heart_with_missing$target)
confusion_matrixxx

#calculate the accuracy of original data set
accuracy_originalll <- sum(diag(confusion_matrixxx))/sum(confusion_matrixxx)
cat("Accuracy for original data:",accuracy_originalll)

#sensitivity
TP <- confusion_matrixxx[2,2] #number of true positive values
FN <- confusion_matrixxx[2,1] #number of false negative values

sensitivity <- TP / (TP + FN) #sensitivity calculation
print(sensitivity)

#specificity
TN <- confusion_matrixxx[1,1] #number of true negative values 
FP <- confusion_matrixxx[1,2] #number of false positive values

specificity <- TN / (TN + FP) #specificity calculation
print(specificity)

#F1-Score
TP <- confusion_matrixxx[2,2] #number of true positive values
FP <- confusion_matrixxx[1,2] #number of false positive values
FN <- confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(f1_score)

#results of Gaussian Naive Bayes classification visualization:
# Creating a dataframe for visualization
metrics_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Value = c(accuracy_originalll, sensitivity, specificity, f1_score)
)

# Creating a bar plot
library(ggplot2)

ggplot(metrics_df, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.5, color = "black") +
  geom_text(aes(label = round(Value, digits = 3)), vjust = -0.5, size = 4) +
  labs(title = "Results of GNB Visualization",
       x = "Metric",
       y = "Value") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  )

library(gplots)
plot(as.table(confusion_matrixxx), main = "GNB Confusion Matrix")

# 6) RANDOM FOREST: Apply the random forest classification method on the data set WITHOUT normalization
library(randomForest)
rf_model <- randomForest(target ~., data = trainDataa)
predictions <- predict(rf_model, newdata = heart_with_missing)

#find the confusion matrix
confusion_matrixxx <- table(predictions,heart_with_missing$target)
confusion_matrixxx

#calculate the accuracy of original data set
accuracy_originalll <- sum(diag(confusion_matrixxx))/sum(confusion_matrixxx)
cat("Accuracy for original data:",accuracy_originalll)

#sensitivity
TP <- confusion_matrixxx[2,2] #number of true positive values
FN <- confusion_matrixxx[2,1] #number of false negative values

sensitivity <- TP / (TP + FN) #sensitivity calculation
print(sensitivity)

#specificity
TN <- confusion_matrixxx[1,1] #number of true negative values 
FP <- confusion_matrixxx[1,2] #number of false positive values

specificity <- TN / (TN + FP) #specificity calculation
print(specificity)

#F1-Score
TP <- confusion_matrixxx[2,2] #number of true positive values
FP <- confusion_matrixxx[1,2] #number of false positive values
FN <- confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(f1_score)


#results of Random Forest classification visualization:
# Creating a dataframe for visualization
metrics_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Value = c(accuracy_originalll, sensitivity, specificity, f1_score)
)

# Creating a bar plot
library(ggplot2)

ggplot(metrics_df, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.5, color = "black") +
  geom_text(aes(label = round(Value, digits = 3)), vjust = -0.5, size = 4) +
  labs(title = "Performance Metrics",
       x = "Metric",
       y = "Value") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  )
library(gplots)
plot(as.table(confusion_matrixxx), main = "Confusion Matrix")

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
  #After the normalization part, apply the all steps for the normalized data frame too
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
# 1) KNN: Apply the KNN algorithm on the normalized data set 
#split the data set into train(70%),test(15%), and validation(15%)
install.packages("caret")
library(caret)

set.seed(123)
trainIndex <- createDataPartition(normalized_df$target, p = 0.7, list = FALSE)
trainDataa <- normalized_df[trainIndex,]
temporaryData <- normalized_df[-trainIndex,]

validationIndex <- createDataPartition(temporaryData$target, p = 0.5, list = FALSE)
validationDataa <- temporaryData[validationIndex,]
testDataa <- temporaryData[-validationIndex,] #real values

#check the number of rows of the splitted data
nrow(trainDataa) 
nrow(testDataa)
nrow(validationDataa)

#apply the knn algorithm
library(class)
knn_modell <- knn(train = trainDataa[,1:13], 
                  test = testDataa[,1:13],
                  cl = trainDataa$target,
                  k = 7)

knn_modell #predicted values
print(c(testDataa$target)) #real values

#find the confusion matrix
confusion_matrixxx <- table(Reference = testDataa$target,  
                            Prediction = knn_modell)
cat("Confusion matrix of original data:")
confusion_matrixxx

#calculate the accuracy of original data set
accuracy_originalll <- sum(diag(confusion_matrixxx))/sum(confusion_matrixxx)
cat("Accuracy for original data:",accuracy_originalll)

#sensitivity
TP <- confusion_matrixxx[2,2] #number of true positive values
FN <- confusion_matrixxx[2,1] #number of false negative values

sensitivity <- TP / (TP + FN) #sensitivity calculation
print(sensitivity)

#specificity
TN <- confusion_matrixxx[1,1] #number of true negative values 
FP <- confusion_matrixxx[1,2] #number of false positive values

specificity <- TN / (TN + FP) #specificity calculation
print(specificity)

#F1-Score
TP <- confusion_matrixxx[2,2] #number of true positive values
FP <- confusion_matrixxx[1,2] #number of false positive values
FN <- confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

f1_score <- (2 * (precision * recall)) / (precision + recall) #f1_score calculation
print(f1_score)


#results of normalized KNN classification visualization:
sensitivity_values <- c(sensitivity)
specificity_values <- c(specificity) 
f1_score_values <- c(f1_score)  

labels <- c("Sensitivity", "Specificity", "f1-Score")

barplot_matrix <- matrix(c(sensitivity_values, specificity_values, f1_score_values), nrow = 3, byrow = TRUE)
barplot(t(barplot_matrix), beside = TRUE, col = c("orange", "purple", "pink"), 
        main = "Results of the normalized KNN", xlab = "Metrics", ylab = "Values",
        names.arg = labels)

library(gplots)
plot(as.table(confusion_matrixxx), main = "Confusion Matrix")

# 2)LOGISTIC REGRESSION: Apply the logistic regression on the normalized data set
independent_var <- normalized_df[, 1:13]
dependent_var <- normalized_df[, 14]

logit_model <- glm(dependent_var ~ ., data = normalized_df, family = binomial) #compose the logistic regression model
#logit_model

predictions <- predict(logit_model, newdata = normalized_df, type="response")
predicted_classes <- ifelse(predictions > 0.5,1,0)

#find the confusion matrix
confusion_matrixxx <- table(predicted_classes,dependent_var)
print(confusion_matrixxx)

#calculate the accuracy of original data set
accuracy_originalll <- confusion_matrixxx['Accuracy']
print(paste("Accuracy:",accuracy_originalll))

#sensitivity
sensitivity <- confusion_matrixxx['Sensitivity']
print(paste("Sensitivity",sensitivity))

#specificity
specificity <- confusion_matrixxx['Specificity']
print(paste("Specificity",specificity))

#F1-Score
f1_score <- confusion_matrixxx['F1']
print(paste("F1 Score",f1_score))


#results of normalized Logistic Regression classification visualization:

library(ggplot2)
# Creating a dataframe for visualization
metrics_df <- data.frame(
  Metric = c("Accuracy","Sensitivity", "Specificity", "F1 Score"),
  Value = c(as.numeric(accuracy),as.numeric(sensitivity), as.numeric(specificity), as.numeric(f1_score))
)

# Creating a bar plot
ggplot(metrics_df, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.5, color = "black") +
  labs(title = "Normalized Logistic Regression Results",
       x = "Metric",
       y = "Value") +
  geom_text(aes(label = sprintf("%.3f", Value)), vjust = -0.5, size = 4) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  )

library(gplots)
plot(as.table(confusion_matrixxx), main = "Normalized Logistic Regression Confusion Matrix")


# 3)DECISION TREE: Apply the decision tree on the normalized data set 
independent_var <- normalized_df[, 1:13]
dependent_var <- normalized_df[, 14]

install.packages("rpart")
library(rpart)
decision_tree <- rpart(dependent_var ~. ,data = independent_var)
#print(decision_tree)
plot(decision_tree)
text(decision_tree)

predictions <- predict(decision_tree, newdata = normalized_df,type="class")

#find the confusion matrix
confusion_matrixxx <- table(predictions,dependent_var)
confusion_matrixxx

#calculate the accuracy of original data set
accuracy_originalll <- sum(diag(confusion_matrixxx))/sum(confusion_matrixxx)
cat("Accuracy for original data:",accuracy_originalll)

#sensitivity
TP <- confusion_matrixxx[2,2] #number of true positive values
FN <- confusion_matrixxx[2,1] #number of false negative values

sensitivity <- TP / (TP + FN) #sensitivity calculation
print(sensitivity)

#specificity
TN <- confusion_matrixxx[1,1] #number of true negative values 
FP <- confusion_matrixxx[1,2] #number of false positive values

specificity <- TN / (TN + FP) #specificity calculation
print(specificity)

#F1-Score
TP <- confusion_matrixxx[2,2] #number of true positive values
FP <- confusion_matrixxx[1,2] #number of false positive values
FN <- confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

f1_score <- (2 * (precision * recall)) / (precision + recall) #f1_score calculation
print(f1_score)

#results of normalized Decision Tree classification visualization:
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
  labs(title = "Normalized Decision Tree Visualization",
       x = "Metric",
       y = "Value") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  )

library(gplots)
plot(as.table(confusion_matrixxx), main = "Normalized Decision Tree Confusion Matrix")


# 4)SVM: Apply the support vector machine on the normalized data set
library(e1071)
set.seed(123)
svm_model <- svm(target ~., data = normalized_df)

predictions <- predict(svm_model,normalized_df)

#confusion matrix
confusion_matrixxx <- table(predictions,normalized_df$target)
print("Confusion matrix:")
print(confusion_matrixxx)


#calculate the accuracy of original data set
accuracy_originalll <- sum(diag(confusion_matrixxx))/sum(confusion_matrixxx)
cat("Accuracy for original data:",accuracy_originalll)

#sensitivity
TP <- confusion_matrixxx[2,2] #number of true positive values
FN <- confusion_matrixxx[2,1] #number of false negative values

sensitivity <- TP / (TP + FN) #sensitivity calculation
print(sensitivity)

#specificity
TN <- confusion_matrixxx[1,1] #number of true negative values 
FP <- confusion_matrixxx[1,2] #number of false positive values

specificity <- TN / (TN + FP) #specificity calculation
print(specificity)

#F1-Score
TP <- confusion_matrixxx[2,2] #number of true positive values
FP <- confusion_matrixxx[1,2] #number of false positive values
FN <- confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(f1_score)


#results of Normalized SVM classification visualization:
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
  labs(title = "Normalized SVM Visualization",
       x = "Metric",
       y = "Value") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  )

library(gplots)
plot(as.table(confusion_matrixxx), main = "Normalized SVM Confusion Matrix")


# 5) GAUSSIAN NAIVE BAYES: Apply the gaussian naive bayes on the normalized data set
library(e1071)
gnb_model <- naiveBayes(target ~., data = trainDataa)
predictions <- predict(gnb_model, newdata = normalized_df)

#find the confusion matrix
confusion_matrixxx <- table(predictions,normalized_df$target)
confusion_matrixxx

#calculate the accuracy of original data set
accuracy_originalll <- sum(diag(confusion_matrixxx))/sum(confusion_matrixxx)
cat("Accuracy for original data:",accuracy_originalll)

#sensitivity
TP <- confusion_matrixxx[2,2] #number of true positive values
FN <- confusion_matrixxx[2,1] #number of false negative values

sensitivity <- TP / (TP + FN) #sensitivity calculation
print(sensitivity)

#specificity
TN <- confusion_matrixxx[1,1] #number of true negative values 
FP <- confusion_matrixxx[1,2] #number of false positive values

specificity <- TN / (TN + FP) #specificity calculation
print(specificity)

#F1-Score
TP <- confusion_matrixxx[2,2] #number of true positive values
FP <- confusion_matrixxx[1,2] #number of false positive values
FN <- confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(f1_score)

#results of Normalized Gaussian Naive Bayes classification visualization:
# Creating a dataframe for visualization
metrics_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Value = c(accuracy_originalll, sensitivity, specificity, f1_score)
)

# Creating a bar plot
library(ggplot2)

ggplot(metrics_df, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.5, color = "black") +
  geom_text(aes(label = round(Value, digits = 3)), vjust = -0.5, size = 4) +
  labs(title = "Results of Normalized GNB Visualization",
       x = "Metric",
       y = "Value") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  )


library(gplots)
plot(as.table(confusion_matrixxx), main = "Normalized GNB Confusion Matrix")


# 6) RANDOM FOREST: Apply the random forest classification method on the normalized data set
library(randomForest)
rf_model <- randomForest(target ~., data = trainDataa)
predictions <- predict(rf_model, newdata = normalized_df)

#find the confusion matrix
confusion_matrixxx <- table(predictions,normalized_df$target)
confusion_matrixxx

#calculate the accuracy of original data set
accuracy_originalll <- sum(diag(confusion_matrixxx))/sum(confusion_matrixxx)
cat("Accuracy for original data:",accuracy_originalll)

#sensitivity
TP <- confusion_matrixxx[2,2] #number of true positive values
FN <- confusion_matrixxx[2,1] #number of false negative values

sensitivity <- TP / (TP + FN) #sensitivity calculation
print(sensitivity)

#specificity
TN <- confusion_matrixxx[1,1] #number of true negative values 
FP <- confusion_matrixxx[1,2] #number of false positive values

specificity <- TN / (TN + FP) #specificity calculation
print(specificity)

#F1-Score
TP <- confusion_matrixxx[2,2] #number of true positive values
FP <- confusion_matrixxx[1,2] #number of false positive values
FN <- confusion_matrixxx[2,1] #number of false negative values
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

f1_score <- (2 * (precision + recall)) / (precision + recall) #f1_score calculation
print(f1_score)

#results of normalized Random Forest classification visualization:
# Creating a dataframe for visualization
metrics_df <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1 Score"),
  Value = c(accuracy_originalll, sensitivity, specificity, f1_score)
)

# Creating a bar plot
library(ggplot2)

ggplot(metrics_df, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.5, color = "black") +
  geom_text(aes(label = round(Value, digits = 3)), vjust = -0.5, size = 4) +
  labs(title = "Results of Normalized Random Forest",
       x = "Metric",
       y = "Value") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none"
  )
library(gplots)
plot(as.table(confusion_matrixxx), main = "Normalized Random Forest Confusion Matrix")

#end of the classification part of the normalized data frame






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
head(normalized_df)

set.seed(123)
trainIndex <- createDataPartition(data$target, p = 0.7, list = FALSE)
trainDataa <- normalized_df[trainIndex,]
temporaryData <- normalized_df[-trainIndex,]

validationIndex <- createDataPartition(temporaryData$target, p = 0.5, list = FALSE)
validationDataa <- temporaryData[validationIndex,]
testDataa <- temporaryData[-validationIndex,] #real values







############

tune_grid <- expand.grid(C = c(0.1,1,10),gamma = c(0.01,0.1,1))
svm_model <- svm(target ~. , data=trainDataa,
                 kernel = "radial",
                 probability = TRUE)

tuned_model<- tune(svm,train.x = trainDataa[,-13],
                   train.y = trainDataa$target,
                   kernel = "radial",
                   ranges = list(C = c(0.1,1,10),
                                 gamma = c(0.01,0.1,1)))

print(tuned_model)

best_model <- tuned_model$best.model

train_predictions <- predict(best_model,
                             newdata= trainDataa[,-13])
train_conf_matrix <- confusionMatrix(train_predictions,trainDataa$target)
print("Train data results:")
print(train_conf_matrix)


best_model <- tuned_model$best.model

test_predictions <- predict(best_model,
                            newdata= testDataa[,-13])
test_conf_matrix <- confusionMatrix(test_predictions,testDataa$target)
print("Test data results:")
print(test_conf_matrix)



# 
# #apply the knn algorithm
# library(class)
# library(caret)
# # knn_modell <- knn(train = trainDataa[,1:13],
# #                   test = testDataa[,1:13],
# #                   cl = trainDataa$target,
# #                   k = 7)
# # 
# # knn_modell #predicted values
# # print(c(testDataa$target)) #real values
# 
# tune_grid <- expand.grid(C = c(0.1,1,10), 
#                          gamma = c(0.01,0.1,1))
# 
# svm_model <- svm(target ~. , data=trainDataa,
#                  kernel = "radial",
#                  probability = TRUE)
# 
# tuned_model <- tune(svm, 
#                     train.x = trainDataa[,-13],
#                     train.y = trainDataa$target,
#                     kernel="radial",
#                     ranges = list( C = c(0.1,1,10),
#                                    gamma = c(0.01,0.1,1)))
# print(tuned_model)
# 
# 
# best_model <- tuned_model$best.model
# 
# train_predictions <- predict(best_model, newdata = trainDataa[,-13])
# 
# train_conf_matrix <- confusionMatrix(train_predictions,trainDataa$target)
# print("Train data results:")
# print(train_conf_matrix)
# 
# test_predictions <- predict(best_model, newdata = testDataa[,-13])
# test_conf_matrix <- confusionMatrix(test_predicitons,testDataa$target)
# print("Train data results:")
# print(test_conf_matrix)






