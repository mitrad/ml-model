library(tidyverse)
library(caret)
library(mlbench)
library(C50)

data("BostonHousing")

glimpse(BostonHousing)

# 1. select independent variable
ind <- dput(names(BostonHousing))
ind_sel <- setdiff(ind, c("medv", "chas"))

# 2. Modeling
housing_formula <- as.formula(
    paste("medv", "~", paste(ind_sel, collapse = "+"), collapse = "")
)

# 3. Model fitting
model <- lm(housing_formula, data = BostonHousing[1:100, ])

# 4. Prediction
predicted <- predict(model, BostonHousing[1:100, ], type = "response")

# 5. RMSE
actual <- BostonHousing[1:100, "medv"]
sqrt(mean((predicted - actual)^2))

# 6. Prediction with test set
predicted <- predict(model, BostonHousing[101:200, ], type = "response")
# 7. RMSE with test set
actual <- BostonHousing[101:200, "medv"]
sqrt(mean((predicted - actual)^2))

# Cross Validation
set.seed(77)
model <- train(housing_formula, BostonHousing,
               method = "lm",
               trControl = trainControl(method = "cv",
                                        number = 10,
                                        verboseIter = TRUE)
)
model

# Split data
idx <- createDataPartition(BostonHousing$medv, p = 0.7, list = FALSE)
caret_train <- BostonHousing[idx, ]
caret_test <- BostonHousing[-idx, ]

nrow(caret_train) / nrow(BostonHousing)
nrow(caret_test) / nrow(BostonHousing)

## Confusion Matrix
data(Sonar)

set.seed(123)
Sonar <- Sonar[sample(nrow(Sonar)), ]
idx <- round(nrow(Sonar) * 0.7)
train <- Sonar[1:idx, ]
test <- Sonar[(idx + 1):nrow(Sonar), ]

model_logit <- glm(Class ~ ., family = "binomial", data = train)

logit_prob <- predict(model_logit, test, type = "response")
logit_prob_class <- ifelse(logit_prob > 0.5, "M", "R") %>%
    as.factor

table(logit_prob_class, test[["Class"]])

confusionMatrix(logit_prob_class, test[["Class"]])

library(caTools)
colAUC(logit_prob, test[["Class"]], plotROC = TRUE)
