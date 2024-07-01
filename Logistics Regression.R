# Function to check and install packages
install_if_missing <- function(packages) {
  new.packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new.packages)) install.packages(new.packages)
  sapply(packages, require, character.only = TRUE)
}

# List of required packages
required_packages <- c("tidyverse", "caret", "pROC", "rpart", "rpart.plot", "car")

# Install and load the required packages
install_if_missing(required_packages)

# Load the dataset
data <- read.csv('C:/Users/nihar/OneDrive/Desktop/Bootcamp/SCMA 632/Assignments/A3/Loan Eligibility Prediction.csv')

# Convert relevant variables to appropriate types
data$Loan_Status <- as.factor(data$Loan_Status)
data$Gender <- as.factor(data$Gender)
data$Married <- as.factor(data$Married)
data$Dependents <- as.factor(data$Dependents)
data$Education <- as.factor(data$Education)
data$Self_Employed <- as.factor(data$Self_Employed)
data$Property_Area <- as.factor(data$Property_Area)

# Recode Loan_Status to binary
data$Loan_Status <- ifelse(data$Loan_Status == "Y", 1, 0)

# Identify and fill missing values
data <- data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)),
         across(where(is.factor), ~ ifelse(is.na(.), as.character(stats::na.omit(.))[which.max(tabulate(match(. , as.character(stats::na.omit(.)))))] , .)))

# Identify and cap outliers using the IQR method
cap_outliers <- function(x) {
  qnt <- quantile(x, probs = c(.25, .75), na.rm = TRUE)
  caps <- quantile(x, probs = c(.05, .95), na.rm = TRUE)
  H <- 1.5 * IQR(x, na.rm = TRUE)
  x[x < (qnt[1] - H)] <- caps[1]
  x[x > (qnt[2] + H)] <- caps[2]
  return(x)
}

data <- data %>%
  mutate(across(where(is.numeric), cap_outliers))

# Validate assumptions for logistic regression
# Check multicollinearity using VIF
logit_model <- glm(Loan_Status ~ ., data = data, family = binomial)
vif_values <- vif(logit_model)
cat("VIF Values:\n")
print(vif_values)

# Check linearity of logit
probabilities <- predict(logit_model, type = "response")
logit_residuals <- residuals(logit_model, type = "deviance")
plot(logit_residuals ~ probabilities, main = "Residuals vs Predicted Probabilities")

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$Loan_Status, p = .8, list = FALSE, times = 1)
train_data <- data[trainIndex,]
test_data  <- data[-trainIndex,]

# Logistic Regression Model
logistic_model <- glm(Loan_Status ~ Gender + Married + Dependents + Education + Self_Employed + Applicant_Income + Coapplicant_Income + Loan_Amount + Loan_Amount_Term + Credit_History + Property_Area, data = train_data, family = binomial)

# Summary of the model
summary(logistic_model)

# Stepwise regression for model selection
stepwise_model <- step(logistic_model, direction = "both")
summary(stepwise_model)

# Final model based on stepwise selection
final_model <- stepwise_model

# Detailed interpretation of significant predictors
cat("Interpreting Significant Predictors:\n")
coefficients <- summary(final_model)$coefficients
for (predictor in rownames(coefficients)) {
  cat(predictor, ":", coefficients[predictor, "Estimate"], "\n")
  if (predictor == "(Intercept)") {
    cat("Interpretation: This is the log odds of loan approval when all predictors are zero.\n")
  } else {
    cat("Interpretation: The effect of", predictor, "on the log odds of loan approval.\n")
  }
}

# Predict on the test data using the final model
pred_prob_final <- predict(final_model, test_data, type = "response")
pred_final <- ifelse(pred_prob_final > 0.5, 1, 0)

# Confusion matrix for final logistic regression model
conf_matrix_final <- table(pred_final, test_data$Loan_Status)
print(conf_matrix_final)

# Calculate accuracy, precision, recall, and F1 score for the final model
accuracy_final <- sum(diag(conf_matrix_final)) / sum(conf_matrix_final)
precision_final <- conf_matrix_final[2,2] / sum(conf_matrix_final[2,])
recall_final <- conf_matrix_final[2,2] / sum(conf_matrix_final[,2])
f1_score_final <- 2 * ((precision_final * recall_final) / (precision_final + recall_final))

# Print metrics for the final logistic regression model
cat("Final Logistic Regression Model Accuracy:", accuracy_final, "\n")
cat("Final Logistic Regression Model Precision:", precision_final, "\n")
cat("Final Logistic Regression Model Recall:", recall_final, "\n")
cat("Final Logistic Regression Model F1 Score:", f1_score_final, "\n")

# ROC curve and AUC for the final logistic regression model
roc_curve_final <- roc(test_data$Loan_Status, as.numeric(pred_prob_final))
plot(roc_curve_final, main = "ROC Curve for Logistic Regression")
auc_value_final <- auc(roc_curve_final)
cat("Final Logistic Regression Model AUC:", auc_value_final, "\n")

# Decision Tree Model
tree_model <- rpart(Loan_Status ~ Gender + Married + Dependents + Education + Self_Employed + Applicant_Income + Coapplicant_Income + Loan_Amount + Loan_Amount_Term + Credit_History + Property_Area, data = train_data, method = "class", control = rpart.control(minsplit = 10, cp = 0.005, maxdepth = 10))

# Plot the decision tree with enhanced visualization
rpart.plot(tree_model, 
           type = 3, 
           extra = 104, 
           fallen.leaves = TRUE, 
           faclen = 0, 
           varlen = 0, 
           box.palette = list("lightblue", "lightgreen"), 
           shadow.col = "gray", 
           main = "Decision Tree for Loan Eligibility Prediction")

# Predict on the test data
tree_pred <- predict(tree_model, test_data, type = "class")

# Confusion matrix for decision tree
tree_conf_matrix <- table(tree_pred, test_data$Loan_Status)
print(tree_conf_matrix)

# Calculate accuracy, precision, recall, and F1 score for decision tree
tree_accuracy <- sum(diag(tree_conf_matrix)) / sum(tree_conf_matrix)
tree_precision <- tree_conf_matrix[2,2] / sum(tree_conf_matrix[2,])
tree_recall <- tree_conf_matrix[2,2] / sum(tree_conf_matrix[,2])
tree_f1_score <- 2 * ((tree_precision * tree_recall) / (tree_precision + tree_recall))

# Print metrics for decision tree
cat("Decision Tree Accuracy:", tree_accuracy, "\n")
cat("Decision Tree Precision:", tree_precision, "\n")
cat("Decision Tree Recall:", tree_recall, "\n")
cat("Decision Tree F1 Score:", tree_f1_score, "\n")

# ROC curve and AUC for decision tree
tree_pred_prob <- predict(tree_model, test_data, type = "prob")[,2]
roc_curve_tree <- roc(test_data$Loan_Status, tree_pred_prob)
plot(roc_curve_tree, main = "ROC Curve for Decision Tree")
auc_value_tree <- auc(roc_curve_tree)
cat("Decision Tree AUC:", auc_value_tree, "\n")

# Compare Logistic Regression and Decision Tree models
comparison <- data.frame(
  Model = c("Logistic Regression", "Decision Tree"),
  Accuracy = c(accuracy_final, tree_accuracy),
  Precision = c(precision_final, tree_precision),
  Recall = c(recall_final, tree_recall),
  F1_Score = c(f1_score_final, tree_f1_score),
  AUC = c(auc_value_final, auc_value_tree)
)

# Print the comparison table
print(comparison)

# Plot the ROC curves for both models for comparison
plot(roc_curve_final, col = "blue", lty = 1, main = "ROC Curves for Logistic Regression and Decision Tree")
lines(roc_curve_tree, col = "red", lty = 2)
legend("bottomright", legend = c("Logistic Regression", "Decision Tree"), col = c("blue", "red"), lty = c(1, 2))

# Save the comparison table to a CSV file
write.csv(comparison, file = "model_comparison.csv", row.names = FALSE)

# Save the ROC plot as an image
png(filename = "ROC_Curves_Comparison.png")
plot(roc_curve_final, col = "blue", lty = 1, main = "ROC Curves for Logistic Regression and Decision Tree")
lines(roc_curve_tree, col = "red", lty = 2)
legend("bottomright", legend = c("Logistic Regression", "Decision Tree"), col = c("blue", "red"), lty = c(1, 2))
dev.off()