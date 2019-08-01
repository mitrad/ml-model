library(caret)
library(tidyverse)
library(yardstick)
library(janitor)
library(caTools)

data("GermanCredit")

credit_df <- GermanCredit %>%
    clean_names() %>%
    tbl_df

gc_model <- glm(class ~ ., data = credit_df, family = "binomial")

gc_pred <- predict(gc_model, type = 'response')

gc_class <- ifelse(gc_pred > 0.5, "Bad", "Good")

table(gc_class, credit_df$class)

###

set.seed(777)

shuffle_index <- sample(nrow(credit_df))
credit_df <- credit_df[shuffle_index, ]

split <- round(nrow(credit_df) * 0.7)

train <- credit_df[1:split, ]
test <- credit_df[(split + 1):nrow(credit_df), ]

gc_model <- glm(class ~ ., data = train, family = "binomial")
gc_pred <- predict(gc_model, data = train, type = "response")

gc_class <- ifelse(gc_pred > 0.5, "Bad", "Good")
table(gc_class, train$class)

###

train_test_index <- createDataPartition(credit_df$class, p = 0.7, list = FALSE)

train <- credit_df[train_test_index, ]
test <- credit_df[-train_test_index, ]

cv_folds <- createMultiFolds(train$class, k = 5, times = 3)

cv_ctrl <- trainControl(method = "cv", number = 5,
                        repeats = 3,
                        index = cv_folds,
                        verboseIter = TRUE)

gc_model <- train(class ~ ., train,
                  method = "glm",
                  family = "binomial",
                  trControl = cv_ctrl)
gc_pred_class <- predict(gc_model, newdata = test, type = "raw")
table(gc_pred_class, test$class)

confusionMatrix(gc_pred_class, test$class)

gc_pred <- predict(gc_model, newdata = test, type = "prob")

colAUC(gc_pred, test[["class"]], plotROC = TRUE, alg = c("ROC"))

###
cv_folds <- createMultiFolds(train$class, k = 10, times = 3)

cv_ctrl <- trainControl(method = "cv", number = 10,
                        repeats = 3,
                        index = cv_folds,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        verboseIter = FALSE)

gc_model <- train(class ~ ., train,
                  method = "glm",
                  family = "binomial",
                  metric = "ROC",
                  trControl = cv_ctrl)

gc_pred_class <- predict(gc_model, newdata = test, type = "raw")
confusionMatrix(gc_pred_class, test$class)

gc_pred <- predict(gc_model, newdata = test, type = "prob")
colAUC(gc_pred, test[["class"]], plotROC = TRUE, alg = c("ROC"))

##
cv_folds <- createMultiFolds(train$class, k = 5, times = 3)
cv_ctrl <- trainControl(method = "cv", number = 5,
                        repeats = 3,
                        index = cv_folds,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        verboseIter = TRUE)
gc_model <- train(class ~., train,
                  method = "ranger",
                  metric = "Sens",
                  tuneLength = 3,
                  trControl = cv_ctrl)
gc_model
plot(gc_model)

gc_pred_class <- predict(gc_model, newdata = test, type = "raw")
confusionMatrix(gc_pred_class, test$class)
gc_pred <- predict(gc_model, newdata = test, type = "prob")
colAUC(gc_pred, test$class, plotROC = TRUE, alg = c("ROC"))

## tune hyper parameter

cv_folds <- createMultiFolds(train$class, k = 5, times = 3)
cv_ctrl <- trainControl(method = "cv", number = 5,
                        repeats = 3,
                        index = cv_folds,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        verboseIter = FALSE)
tune_grid <- data.frame(
    .mtry = c(2, 16, 31, 46),
    .splitrule = c("gini", "extratrees"),
    .min.node.size = c(1, 10)
)

gc_model <- train(class ~ ., train,
                  method = "ranger",
                  metric = "Sens",
                  tuneGrid = tune_grid,
                  trControl = cv_ctrl)
gc_model
plot(gc_model)

gc_pred_class <- predict(gc_model, newdata = test, type = "raw")
confusionMatrix(gc_pred_class, test$class)
gc_pred <- predict(gc_model, newdata = test, type = "prob")
colAUC(gc_pred, test$class, plotROC = TRUE, alg = c("ROC"))

##
credit_var <- setdiff(colnames(credit_df), list('class'))
credit_formula <- as.formula(paste("class", paste(credit_var, collapse = ' + '),
                                   sep = '~'))

cv_folds <- createMultiFolds(train$class, k = 5, times = 3)
cv_ctrl <- trainControl(method = "cv", number = 5,
                        repeats = 3,
                        index = cv_folds,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        verboseIter = FALSE)
tune_grid <- expand.grid(
    alpha = 0:1,
    lambda = seq(0.0001, 1, length = 10)
)

gc_model <- train(credit_formula, train,
                  method = "glmnet",
                  metric = "Sens",
                  tuneGrid = tune_grid,
                  trControl = cv_ctrl)

plot(gc_model)

gc_pred_class <- predict(gc_model, newdata = test, type = "raw")
confusionMatrix(gc_pred_class, test$class)
gc_pred <- predict(gc_model, newdata = test, type = "prob")
colAUC(gc_pred, test$class, plotROC = TRUE, alg = c("ROC"))

##
start_time <- Sys.time()

cv_folds <- createMultiFolds(train$class, k = 5, times = 3)
cv_ctrl <- trainControl(method = "cv", number = 5,
                        repeats = 3,
                        index = cv_folds,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        verboseIter = FALSE)

ranger_tune_grid <- data.frame(
    .mtry = c(2, 16, 31, 46),
    .splitrule = c("gini", "extratrees"),
    .min.node.size = c(5, 10)
)

glmnet_tune_grid <- expand.grid(
    alpha = 0:1,
    lambda = seq(0.0001, 1, length = 10)
)


gc_ranger_model <- train(class ~ ., train,
                         method = "ranger",
                         metric = "Sens",
                         tuneGrid = ranger_tune_grid,
                         trControl = cv_ctrl)

gc_glmnet_model <- train(class ~ ., train,
                         method = "glmnet",
                         metric = "Sens",
                         tuneGrid = glmnet_tune_grid,
                         trControl = cv_ctrl)

model_list <- list(
    glmnet = gc_glmnet_model,
    ranger = gc_ranger_model
)

resamps <- resamples(model_list)
summary(resamps)
dotplot(resamps, metric = "Sens")

gc_pred_class <- predict(gc_glmnet_model, newdata = test, type = "raw")
confusionMatrix(gc_pred_class, test$class)
total_time <- Sys.time() - start_time

diff_test <- resamples(model_list)
diff(diff_test) %>% summary

##
start_time <- Sys.time()

cv_folds <- createMultiFolds(train$class, k = 5, times = 3)
cv_ctrl <- trainControl(method = "cv", number = 5,
                        repeats = 3,
                        index = cv_folds,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        verboseIter = FALSE)

ranger_tune_grid <- data.frame(
    .mtry = c(2, 16, 31, 46),
    .splitrule = c("gini", "extratrees"),
    .min.node.size = c(5, 10)
)

glmnet_tune_grid <- expand.grid(
    alpha = 0:1,
    lambda = seq(0.0001, 1, length = 10)
)


gc_ranger_model <- train(class ~ ., train,
                         method = "ranger",
                         metric = "Sens",
                         preProcess = c("nzv", "center", "scale", "pca"),
                         #tuneGrid = ranger_tune_grid,
                         tuneLength = 7,
                         trControl = cv_ctrl)

gc_glmnet_model <- train(class ~ ., train,
                         method = "glmnet",
                         metric = "Sens",
                         preProcess = c("nzv", "center", "scale", "spatialSign"),
                         tuneGrid = glmnet_tune_grid,
                         trControl = cv_ctrl)

model_list <- list(
    glmnet = gc_glmnet_model,
    ranger = gc_ranger_model
)

resamps <- resamples(model_list)
summary(resamps)
dotplot(resamps, metric = "Sens")

gc_pred_class <- predict(gc_glmnet_model, newdata = test, type = "raw")
confusionMatrix(gc_pred_class, test$class)
total_time <- Sys.time() - start_time

diff_test <- resamples(model_list)
diff(diff_test) %>% summary

##
library(doSNOW)

num_cores <- parallel::detectCores()
cl <- makeCluster(num_cores, type = "SOCK")
registerDoSNOW(cl)

start_time <- Sys.time()
cv_folds <- createMultiFolds(train$class, k = 5, times = 3)
cv_ctrl <- trainControl(method = "cv", number = 5,
                        repeats = 3,
                        index = cv_folds,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        savePredictions = TRUE,
                        verboseIter = FALSE)

ranger_tune_grid <- data.frame(
    .mtry = c(2, 16, 31, 46),
    .splitrule = c("gini", "extratrees"),
    .min.node.size = c(5, 10)
)

glmnet_tune_grid <- expand.grid(
    alpha = 0:1,
    lambda = seq(0.0001, 1, length = 10)
)


gc_ranger_model <- train(class ~ ., train,
                         method = "ranger",
                         metric = "Sens",
                         preProcess = c("nzv", "center", "scale", "pca"),
                         #tuneGrid = ranger_tune_grid,
                         tuneLength = 7,
                         trControl = cv_ctrl)

gc_glmnet_model <- train(class ~ ., train,
                         method = "glmnet",
                         metric = "Sens",
                         preProcess = c("nzv", "center", "scale", "spatialSign"),
                         tuneGrid = glmnet_tune_grid,
                         trControl = cv_ctrl)

model_list <- list(
    glmnet = gc_glmnet_model,
    rf = gc_ranger_model
)

resamps <- resamples(model_list)
summary(resamps)
dotplot(resamps, metric = "Sens")

gc_pred_class <- predict(gc_glmnet_model, newdata = test, type = "raw")
confusionMatrix(gc_pred_class, test$class)
total_time <- Sys.time() - start_time
