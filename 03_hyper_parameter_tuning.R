library(tidyverse)
library(caret)
library(tictoc)
library(janitor)
library(doSNOW)

data("GermanCredit")

credit_dat <- GermanCredit %>%
    clean_names() %>%
    tbl_df

all_variables <- names(credit_dat)
remove_variable <- names(credit_dat)[nearZeroVar(credit_dat)]
# nearZeroVar() 정보량이 적은 변수 추출

credit_df <- credit_dat[, setdiff(all_variables, remove_variable)]

train_test_index <- createDataPartition(credit_df$class, p = 0.7, list = F)
train <- credit_df[train_test_index, ]
test <- credit_df[-train_test_index, ]

# grid search
num_cores <- parallel::detectCores()
cl <- makeCluster(num_cores, type = "SOCK")
registerDoSNOW(cl)

tic()

cv_folds <- createMultiFolds(train$class, k = 3, times = 1)
fit_ctrl <- trainControl(method = "repeatedcv",
                         number = 3,
                         repeats = 1,
                         index = cv_folds,
                         summaryFunction = twoClassSummary,
                         classProbs = TRUE,
                         verboseIter = TRUE)
ranger_tune_grid <- expand.grid(
    .mtry = c(2, 25, 48),
    .splitrule = c("gini", "extratrees"),
    .min.node.size = 10
)

gc_grid_ranger_model <- train(class ~ ., train,
                              method = "ranger",
                              metric = "Sens",
                              preProcess = c("zv", "center", "scale", "spatialSign"),
                              tuneGrid = ranger_tune_grid,
                              trControl = fit_ctrl)
toc()

gc_grid_ranger_model

# random search
tic()

cv_folds <- createMultiFolds(train$class, k = 3, times = 1)
fit_ctrl <- trainControl(method = "repeatedcv",
                         number = 3,
                         repeats = 1,
                         index = cv_folds,
                         search = "random",
                         summaryFunction = twoClassSummary,
                         classProbs = TRUE,
                         verboseIter = TRUE)

gc_ranger_model <- train(class ~ ., train,
                              method = "ranger",
                              metric = "Sens",
                              preProcess = c("zv", "center", "scale", "spatialSign"),
                              tuneLength = 7,
                              trControl = fit_ctrl)
toc()

gc_ranger_model

# adaptive resampling
tic()

cv_folds <- createMultiFolds(train$class, k = 3, times = 1)
fit_ctrl <- trainControl(method = "repeatedcv",
                         number = 3,
                         repeats = 1,
                         index = cv_folds,
                         search = "random",
                         adaptive = list(min = 3, alpha = 0.05, method = "BT",
                                         complete = F),
                         summaryFunction = twoClassSummary,
                         classProbs = TRUE,
                         verboseIter = TRUE)

gc_ranger_model <- train(class ~ ., train,
                         method = "ranger",
                         metric = "Sens",
                         preProcess = c("zv", "center", "scale", "spatialSign"),
                         tuneLength = 7,
                         trControl = fit_ctrl)
toc()

ggplot(gc_ranger_model)
plot(gc_grid_ranger_model,
#     metric = "Kappa",
     plotType = "level")
