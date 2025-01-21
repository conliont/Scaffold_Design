# Change accordingly to your path
setwd("C:\\Users\\konst\\insybio\\OsteoNet\\bone porosity")

# Load necessary libraries
library(glmnet)
library(dplyr)
library(groupdata2)
library(Hmisc)
library(ggplot2)
library(corrplot)

# Load the datasets
inputs <- read.csv("porosity_inputs.csv", header = TRUE, row.names = 1)
outputs <- read.csv("porosity_outputs.csv", header = TRUE, row.names = 1)

# Remove unwanted columns
#inputs <- scaffolds %>% select(-c(Binary))

# Scale the input data
inputs_matrix<-scale(as.matrix(inputs,nrow=length(rownames(inputs)),ncol=length(colnames(inputs))))
# Scale the input data

# Scale the output data
outputs_matrix<-scale(as.matrix(outputs,nrow=length(rownames(outputs)),ncol=length(colnames(outputs))))

# Calculate scaling parameters for inputs and outputs
input_means <- attr(inputs_matrix, "scaled:center")
input_sds <- attr(inputs_matrix, "scaled:scale")
output_means <- attr(outputs_matrix, "scaled:center")
output_sds <- attr(outputs_matrix, "scaled:scale")

# Save scaling parameters to a CSV file
scaling_params <- data.frame(
  Feature = c(colnames(inputs), "output"), 
  Mean = c(input_means, output_means),
  StdDev = c(input_sds, output_sds)
)
write.csv(scaling_params, "porosity_scaling_parameters.csv", row.names = FALSE)



scaled_data <- data.frame(ID = row.names(inputs), inputs_matrix)

# Add the outcomes and binary columns
scaled_data$Bone.porosity <- outputs$Bone.porosity
#threshold_value <- median(outputs$number.of.cells, na.rm = TRUE)
#scaled_data$Binary <- scaffolds$Binary

# Save the scaled data to CSV in the required format
write.csv(scaled_data, "porosity_scaled.csv", row.names = FALSE)

#outputs_matrix<-as.matrix(outputs,nrow=length(rownames(outputs)),ncol=length(colnames(outputs)))

# Set alpha for Elastic Net (e.g., 0.5 for equal mix of L1 and L2 penalties)
alpha_val <- 0.5

# Fitting the model
#model<-glmnet(inputs_matrix, outputs_matrix, family = "gaussian", alpha = alpha_val)

# Train the model for Bone Porosity
y <- outputs$Bone.porosity
model <- glmnet(inputs_matrix, outputs_matrix, family = "gaussian", alpha = alpha_val)

# Save the model
saveRDS(model, "model_bone_porosity.rds")
cat("Model for Bone Porosity saved as model_bone_porosity.rds\n")

# Plot Regularization Path with Legends
lambda_values <- model$lambda
coefficients <- as.matrix(coef(model))

# Remove the intercept row for better visualization
coefficients <- coefficients[-1, ]

# Plotting the coefficients as a function of lambda
matplot(log(lambda_values), t(coefficients), type = "l", lty = 1, col = 1:ncol(coefficients),
        xlab = "Log Lambda", ylab = "Coefficients", main = "Regularization Path")
legend("topright", legend = rownames(coefficients), col = 1:ncol(coefficients), lty = 1, cex = 0.7)

# Cross-validation
set.seed(1)
cvmodel <- cv.glmnet(inputs_matrix, outputs_matrix, family = "gaussian",alpha = alpha_val,nfolds = 5)
plot(cvmodel)
lambda.min_inputs<-cvmodel$lambda.min
# After finding lambda.min in your cv.glmnet function
lambda.min_inputs <- cvmodel$lambda.min

# Save lambda.min to a CSV file
write.csv(data.frame(Model = "bone_porosity", LambdaMin = lambda.min_inputs), "bone_lambda.csv", row.names = FALSE)

coef(model, s = cvmodel$lambda.min)

# Save the cross-validated model
saveRDS(cvmodel, "cvmodel_bone_porosity.rds")
cat("Cross-validated model saved as cvmodel_bone_porosity.rds\n")

# Extract coefficients at the optimal lambda
coef_inputs <- coef(model, s = lambda.min_inputs)

# Convert coefficients to data frame
coef_df <- as.data.frame(as.matrix(coef_inputs))
coef_df <- coef_df %>% mutate(Feature = rownames(coef_df))
colnames(coef_df) <- c("Coefficient", "Feature")

# Print summary of regression coefficients
cat("Regression Coefficients Summary:\n")
print(coef_df)

# Calculate and plot correlation matrix for inputs
input_cor_matrix <- cor(inputs_matrix)
corrplot(input_cor_matrix, method = "circle", type = "upper", 
         tl.col = "black", tl.cex = 0.8, number.cex = 0.7,
         title = "Correlation Matrix of Inputs", addCoef.col = "black")

# Read the scaled data for further processing
miRData <- read.csv("porosity_scaled.csv", header = TRUE)

newData<-as.data.frame(miRData[,2:5])
d2=miRData[,6]
d1=as.factor(miRData[,1])
#d3=as.factor(miRData[, 7])

#newinputs <- cbind(d1, newData, d2, d3)
newinputs <- cbind(d1, newData, d2)
col1<-colnames(newinputs)
col1[1]<-"ID"
col1[6]<-"Endpnt"
#col1[7] <- "Binary"

colnames(newinputs)<-col1

# Check the data types
str(miRData)
head(miRData)

# Set seed for reproducibility
set.seed(7)

# Number of folds for cross-validation
kfold <- 5
bootCount = NULL

# Stratified Fold to ensure a given patient data in same fold
#ata <- fold(newinputs, k = kfold, cat_col = 'Binary', id_col = 'ID', method = "n_dist") %>% arrange(.folds)
data <- fold(newinputs, k = kfold, id_col = 'ID', method = "n_dist") %>% arrange(.folds)

PTOT <- NULL
Out <- NULL

foldROC <- list()
r1<-rownames(data)
dat2<-as.data.frame(matrix(ncol=5,nrow=0))

dat2[1,1]<-"MODEL"
dat2[1,2]<-0
dat2[1,3]<-0
dat2[1,4]<-0
dat2[1,5]<-0

for (i in 1:kfold) {
  print(i)
  DataC1 <- data[data$.folds == i, ]  # Data that will be predicted
  DataCV <- data[data$.folds != i, ]  # To train the model
  message <- paste("Endpnt~", colnames(data)[2])
  
  for(j in 3:(ncol(data) - 2)){
    message <- paste(message, "+", colnames(data)[j])
  }
  
  print(message)
  print("test11")
  M1 <- glmnet(x = as.matrix(DataCV[, 2:(ncol(DataCV) - 2)]), y = as.matrix(DataCV$Endpnt), family = "gaussian", alpha = alpha_val, type.measure = "mse")
  print("test1")
  #P1 <- predict(M1, newx = as.matrix(DataC1[, 2:(ncol(DataC1) - 2)]), allow.new.levels=TRUE, type = "response")
  # Example after predicting in cross-validation
  P1 <- predict(M1, newx = as.matrix(DataC1[, 2:(ncol(DataC1) - 2)]), s = lambda.min_inputs, allow.new.levels=TRUE, type = "response")
  P1 <- P1[, length(colnames(P1))]
  print("test2")
  names(P1) <- DataC1$ID
  PTOT <- c(PTOT, P1)
  Out <- c(Out, DataC1$Endpnt)
}


# Save the predictions
df <- as.data.frame(do.call(rbind, lapply(PTOT, as.vector)))
df$ID <- rownames(df)
colnames(df) <- c("Predicted", "ID")
write.csv(df, file = "porosity_predictions.csv", row.names = FALSE)

# Merge predictions with actual outputs
outputs$ID <- rownames(outputs)
combined_data <- merge(df, outputs, by = "ID", all.y = TRUE) # Ensure all rows from outputs are kept
combined_data <- combined_data[match(outputs$ID, combined_data$ID), ] # Reorder to match outputs

# Save the combined data to CSV
write.csv(combined_data, "actual_predicted_outputs.csv", row.names = FALSE)

# Calculate and print correlation and p-values
res <- rcorr(as.matrix(PTOT), as.matrix(Out), type = "spearman")
corr <- signif(res$r[1, 2], 3)
pvals <- signif(res[3]$P[1, 2], 3)
mse <- mean((PTOT - Out)^2)
mae <- mean(abs(PTOT - Out))
rmse <- sqrt(mse)
r2 <- 1 - sum((PTOT - Out)^2) / sum((Out - mean(Out))^2)

# Print metrics
cat("Model Evaluation Metrics:\n")
cat(sprintf("Spearman's Correlation: %0.3f\n", corr))
cat(sprintf("Spearman's p-value: %0.3g\n", pvals))
cat(sprintf("Mean Absolute Error (MAE): %0.2f\n", mae))
cat(sprintf("Mean Squared Error (MSE): %0.2f\n", mse))
cat(sprintf("Root Mean Squared Error (RMSE): %0.2f\n", rmse))
cat(sprintf("R-squared (R2): %0.3f\n", r2))

# Plot predicted vs actual values for all folds
ggplot(data.frame(Out, PTOT), aes(x = Out, y = PTOT)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Predicted vs Actual Values", x = "Actual Values", y = "Predicted Values") +
  theme_minimal()

# Calculate residuals
residuals <- Out - PTOT

# Plot residuals
ggplot(data.frame(Out, residuals), aes(x = Out, y = residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residual Plot", x = "Actual Values", y = "Residuals") +
  theme_minimal()

# Extract and plot regression coefficients
coefficients <- coef(model, s = lambda.min_inputs)
coefficients_df <- as.data.frame(as.matrix(coefficients))
coefficients_df$Feature <- rownames(coefficients_df)
colnames(coefficients_df) <- c("Coefficient", "Feature")

# Remove the intercept for plotting
coefficients_df <- coefficients_df[coefficients_df$Feature != "(Intercept)", ]

# Plotting the coefficients with values
ggplot(coefficients_df, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Coefficient, 2)), hjust = -0.2) +
  coord_flip() +
  labs(title = "Regression Coefficients", x = "Features", y = "Coefficient") +
  theme_minimal()

# Create scatter plots for each feature vs. actual and predicted outcomes separately
features <- colnames(inputs)
plot_list_actual <- list()
plot_list_predicted <- list()

for (feature in features) {
  # Calculate Spearman correlation for actual and predicted outcomes
  spearman_corr_actual <- cor(inputs[[feature]], scaled_data$Bone.porosity, method = "spearman")
  spearman_corr_predicted <- cor(inputs[[feature]], combined_data$Predicted, method = "spearman")
  
  actual_plot <- ggplot(data.frame(x = inputs[[feature]], y = scaled_data$Bone.porosity), aes(x = x, y = y)) +
    geom_point(color = "blue") +
    geom_smooth(method = "lm", se = FALSE, color = "blue") +
    labs(title = paste("[ Actual Outcome vs", feature," ]", "\nSpearman correlation::", round(spearman_corr_actual, 3)), 
         x = feature, y = "Actual Outcome") +
    theme_minimal()
  
  predicted_plot <- ggplot(data.frame(x = inputs[[feature]], y = combined_data$Predicted), aes(x = x, y = y)) +
    geom_point(color = "red") +
    geom_smooth(method = "lm", se = FALSE, color = "red") +
    labs(title = paste("[ Predicted Outcome vs", feature," ]", "\nspearman correlation:", round(spearman_corr_predicted, 3)), 
         x = feature, y = "Predicted Outcome") +
    theme_minimal()
  
  plot_list_actual[[feature]] <- actual_plot
  plot_list_predicted[[feature]] <- predicted_plot
}

# Print the actual outcome plots
for (p in plot_list_actual) {
  print(p)
}

# Print the predicted outcome plots
for (p in plot_list_predicted) {
  print(p)
}
