setwd("C:\\Users\\konst\\insybio\\OsteoNet\\testing")

library(glmnet)
library(dplyr)
library(ggplot2)

# === Load model and parameters ===
cvmodel <- readRDS("cvmodel_trabecular_spacing.rds")
scaling_params <- read.csv("spacing_scaling_parameters.csv")
best_lambda <- cvmodel$lambda.min

# Separate scaling parameters
input_scaling <- scaling_params[scaling_params$Feature != "output", ]
output_scaling <- scaling_params[scaling_params$Feature == "output", ]

numerical_means <- input_scaling$Mean
numerical_sds <- input_scaling$StdDev
names(numerical_means) <- input_scaling$Feature
names(numerical_sds) <- input_scaling$Feature
output_mean <- as.numeric(output_scaling$Mean)
output_sd <- as.numeric(output_scaling$StdDev)

# === Load unseen data ===
inputs_unseen <- read.csv("testing_inputs.csv", header = TRUE, row.names = 1)
outputs_unseen <- read.csv("spacing_unseen_labels.csv", header = TRUE, row.names = 1)

# === Scale inputs ===
numerical_inputs <- select(inputs_unseen, all_of(names(numerical_means)))
scaled_inputs <- scale(numerical_inputs, center = numerical_means, scale = numerical_sds)
inputs_matrix_unseen <- as.matrix(scaled_inputs)
actuals <- outputs_unseen$Trabecular.spacing

# === Bootstrapping Setup ===
set.seed(42)
n_iterations <- 1000
n_samples <- floor(0.9 * nrow(inputs_matrix_unseen))

metrics <- data.frame(R2 = numeric(n_iterations),
                      SpearmanRho = numeric(n_iterations),
                      SpearmanP = numeric(n_iterations),
                      MAE = numeric(n_iterations),
                      RMSE = numeric(n_iterations))

for (i in 1:n_iterations) {
  idx <- sample(1:nrow(inputs_matrix_unseen), size = n_samples, replace = TRUE)
  X_boot <- inputs_matrix_unseen[idx, , drop = FALSE]
  y_boot <- actuals[idx]
  
  pred_boot <- predict(cvmodel, newx = X_boot, s = best_lambda)
  pred_boot <- as.numeric(pred_boot)
  
  r2 <- 1 - sum((pred_boot - y_boot)^2) / sum((y_boot - mean(y_boot))^2)
  spearman <- cor.test(pred_boot, y_boot, method = "spearman")
  mae <- mean(abs(pred_boot - y_boot))
  rmse <- sqrt(mean((pred_boot - y_boot)^2))
  
  metrics[i, ] <- c(r2, spearman$estimate, spearman$p.value, mae, rmse)
}

# === Aggregate Results ===
bootstrap_summary <- data.frame(
  Metric = c("R²", "Spearman Correlation", "Spearman p-value", "MAE", "RMSE"),
  Mean = colMeans(metrics),
  StdDev = apply(metrics, 2, sd)
)

write.csv(bootstrap_summary, "bootstrap_spacing_metrics.csv", row.names = FALSE)
cat("Bootstrap evaluation completed. Results saved to 'bootstrap_test_metrics.csv'\n")
print(bootstrap_summary)
