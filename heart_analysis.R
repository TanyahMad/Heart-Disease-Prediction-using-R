library(tidyverse)
library(corrplot)
library(e1071)
library(caret)
library(randomForest)
library(gbm)

# 1. Load data ---------------------------------------------

heartData <- read.csv("synthetic_heart_disease_dataset.csv")

# Convert categorical variables and target to factor
heartData <- heartData %>%
  mutate(
    Gender            = as.factor(Gender),
    Smoking           = as.factor(Smoking),
    Alcohol_Intake    = as.factor(Alcohol_Intake),
    Physical_Activity = as.factor(Physical_Activity),
    Diet              = as.factor(Diet),
    Stress_Level      = as.factor(Stress_Level),
    Heart_Disease     = as.factor(Heart_Disease)   # target
  )

str(heartData)
summary(heartData)
table(heartData$Heart_Disease)

# 2. Exploratory Data Analysis (Visualization) -------------

# helper to force display
show_plot <- function(p) {
  print(p)
  Sys.sleep(0.3)   # pause so RStudio captures it
}

## 2.1 Target (Heart_Disease) class balance ----------------
p1 <- ggplot(heartData, aes(x = Heart_Disease)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Heart Disease Count (0 = No, 1 = Yes)",
       x = "Heart Disease", y = "Number of Patients")
show_plot(p1)

## 2.2 Age distribution ------------------------------------
p2 <- ggplot(heartData, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  labs(title = "Age Distribution", x = "Age", y = "Count")
show_plot(p2)

## 2.3 BMI distribution ------------------------------------
p3 <- ggplot(heartData, aes(x = BMI)) +
  geom_histogram(binwidth = 2, fill = "orange", color = "black") +
  labs(title = "BMI Distribution", x = "BMI", y = "Count")
show_plot(p3)

## 2.4 Total Cholesterol distribution ----------------------
p4 <- ggplot(heartData, aes(x = Cholesterol_Total)) +
  geom_histogram(binwidth = 10, fill = "darkseagreen3", color = "black") +
  labs(title = "Total Cholesterol Distribution",
       x = "Total Cholesterol", y = "Count")
show_plot(p4)

## 2.5 Age vs Heart Disease --------------------------------
p5 <- ggplot(heartData, aes(x = Heart_Disease, y = Age)) +
  geom_boxplot(fill = "violet") +
  labs(title = "Age by Heart Disease Status",
       x = "Heart Disease", y = "Age")
show_plot(p5)

## 2.6 Cholesterol vs Heart Disease ------------------------
p6 <- ggplot(heartData, aes(x = Heart_Disease, y = Cholesterol_Total)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Cholesterol by Heart Disease Status",
       x = "Heart Disease", y = "Cholesterol")
show_plot(p6)

## 2.7 Systolic BP distribution ----------------------------
p7 <- ggplot(heartData, aes(x = Systolic_BP, fill = Heart_Disease)) +
  geom_density(alpha = 0.4) +
  labs(title = "Systolic BP Distribution by Heart Disease")
show_plot(p7)

## 2.8 Smoking vs Heart Disease ----------------------------
p8 <- ggplot(heartData, aes(x = Smoking, fill = Heart_Disease)) +
  geom_bar(position = "fill") +
  labs(title = "Smoking vs Heart Disease", y = "Proportion")
show_plot(p8)

## 2.9 Gender vs Heart Disease -----------------------------
p9 <- ggplot(heartData, aes(x = Gender, fill = Heart_Disease)) +
  geom_bar(position = "fill") +
  labs(title = "Gender vs Heart Disease", y = "Proportion")
show_plot(p9)

## 2.10 Physical Activity vs Heart Disease -----------------
p10 <- ggplot(heartData, aes(x = Physical_Activity, fill = Heart_Disease)) +
  geom_bar(position = "fill") +
  labs(title = "Physical Activity vs Heart Disease", y = "Proportion") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
show_plot(p10)

## 2.11 Stress Level vs Heart Disease ----------------------
p11 <- ggplot(heartData, aes(x = Stress_Level, fill = Heart_Disease)) +
  geom_bar(position = "fill") +
  labs(title = "Stress Level vs Heart Disease", y = "Proportion")
show_plot(p11)

## 2.12 Systolic vs Diastolic BP scatter -------------------
p12 <- ggplot(heartData, aes(x = Systolic_BP, y = Diastolic_BP,
                             color = Heart_Disease)) +
  geom_point(alpha = 0.3) +
  labs(title = "Blood Pressure: Systolic vs Diastolic")
show_plot(p12)

## 2.13 Correlation heatmap --------------------------------
numeric_cols <- sapply(heartData, is.numeric)
numericData <- heartData[, numeric_cols]
corMatrix <- cor(numericData)
corrplot(corMatrix, method = "color", tl.cex = 0.7)

# 3. Train/Test Split --------------------------------------

set.seed(123)
trainIndex <- createDataPartition(heartData$Heart_Disease,
                                  p = 0.7, list = FALSE)
trainData <- heartData[trainIndex, ]
testData  <- heartData[-trainIndex, ]

# 4. Helper function to evaluate models --------------------

evaluate_model <- function(true_labels, predicted_labels, model_name = "Model") {
  true_labels <- factor(true_labels)
  predicted_labels <- factor(predicted_labels, levels = levels(true_labels))
  
  confMat <- table(Actual = true_labels, Predicted = predicted_labels)
  print(paste("Confusion Matrix for", model_name))
  print(confMat)
  
  TP <- confMat[2, 2]  # actual 1, predicted 1
  TN <- confMat[1, 1]  # actual 0, predicted 0
  FP <- confMat[1, 2]
  FN <- confMat[2, 1]
  
  accuracy  <- (TP + TN) / (TP + TN + FP + FN)
  precision <- TP / (TP + FP)
  recall    <- TP / (TP + FN)
  f1_score  <- 2 * (precision * recall) / (precision + recall)
  
  cat("\n", model_name, "metrics:\n")
  cat("Accuracy :", round(accuracy, 3), "\n")
  cat("Precision:", round(precision, 3), "\n")
  cat("Recall   :", round(recall, 3), "\n")
  cat("F1 Score :", round(f1_score, 3), "\n\n")
  
  list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1 = f1_score
  )
}

# 5. Logistic Regression -----------------------------------

logistic_model <- glm(Heart_Disease ~ ., data = trainData, family = binomial)
summary(logistic_model)

logit_prob <- predict(logistic_model, newdata = testData, type = "response")
logit_pred <- ifelse(logit_prob >= 0.5, "1", "0")
logit_metrics <- evaluate_model(testData$Heart_Disease, logit_pred,
                                "Logistic Regression")

# 6. Random Forest -----------------------------------------

set.seed(123)
rf_model <- randomForest(Heart_Disease ~ ., data = trainData,
                         ntree = 300, mtry = 4, importance = TRUE)
print(rf_model)
varImpPlot(rf_model, main = "Random Forest - Variable Importance")

rf_pred <- predict(rf_model, newdata = testData)
rf_metrics <- evaluate_model(testData$Heart_Disease, rf_pred,
                             "Random Forest")

# 7. SVM (linear) -----------------------------------------

set.seed(123)
svm_model <- svm(Heart_Disease ~ ., data = trainData,
                 kernel = "linear", type = "C-classification",
                 cost = 1)
svm_pred <- predict(svm_model, newdata = testData)
svm_metrics <- evaluate_model(testData$Heart_Disease, svm_pred,
                              "SVM (linear)")

# 8. GBM ---------------------------------------------------

gbmTrain <- trainData
gbmTrain$Heart_Disease <- as.numeric(as.character(gbmTrain$Heart_Disease))

set.seed(123)
gbm_model <- gbm(Heart_Disease ~ .,
                 data = gbmTrain,
                 distribution = "bernoulli",
                 n.trees = 500,
                 interaction.depth = 3,
                 shrinkage = 0.01,
                 n.minobsinnode = 10,
                 verbose = FALSE)

gbm_probs <- predict(gbm_model, newdata = testData, n.trees = 500,
                     type = "response")
gbm_pred <- ifelse(gbm_probs >= 0.5, "1", "0")
gbm_metrics <- evaluate_model(testData$Heart_Disease, gbm_pred,
                              "GBM")

# 9. Compare model accuracies ------------------------------

all_accuracies <- tibble(
  Model     = c("Logistic Regression", "Random Forest", "SVM (linear)", "GBM"),
  Accuracy  = c(logit_metrics$accuracy,
                rf_metrics$accuracy,
                svm_metrics$accuracy,
                gbm_metrics$accuracy)
)

print(all_accuracies)

ggplot(all_accuracies, aes(x = Model, y = Accuracy)) +
  geom_col() +
  ylim(0, 1) +
  labs(title = "Model Accuracy Comparison",
       x = "Model", y = "Accuracy")

# 10. Save best model (Random Forest) ----------------------

saveRDS(rf_model, "heart_rf_model.rds")
cat("Random Forest model saved to heart_rf_model.rds\n")

