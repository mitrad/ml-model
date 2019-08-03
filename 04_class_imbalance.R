library(ROSE)
library(tidyverse)
library(rpart)
library(caret)
library(plotROC)
library(ggpubr)
library(extrafont)
loadfonts()

data(hacide)

hacide.train <- hacide.train %>%
    mutate(cls = factor(cls, labels = c("no", "yes")))

hacide.test <- hacide.test %>%
    mutate(cls = factor(cls, label = c("no", "yes")))

hacide.train %>%
    ggplot(aes(x = x1, y = x2, color = cls)) +
    geom_point() +
    theme_pubr(base_family = "NanumGothic") +
    theme(legend.position = "top") +
    labs(color = "종속변수(cls)") +
    scale_color_manual(values = c("lightblue", "red"))

hacide.train %>%
    count(cls) %>%
    mutate(rate = scales::percent(n / sum(n)))

##
balanced_over_sampling_df <- ovun.sample(cls ~. ,data = hacide.train,
                                         method = "over", N = 1960)$data
balanced_under_sampling_df <- ovun.sample(cls ~ ., data = hacide.train,
                                          method = "under", N = 40, seed = 1)$data
balanced_both_sampling_df <- ovun.sample(cls ~ ., data = hacide.train,
                                         method = "both", N = 1000, seed = 1)$data
rose_df <- ROSE(cls ~ ., data = hacide.train)$data

##
raw_rpart <- rpart(cls ~ ., data = hacide.train)
over_rpart <- rpart(cls ~ ., data = balanced_over_sampling_df)
under_rpart <- rpart(cls ~., data = balanced_under_sampling_df)
both_rpart <- rpart(cls ~., data = balanced_both_sampling_df)
rose_rpart <- rpart(cls ~., data = rose_df)

pred_raw_rpart <- predict(raw_rpart, newdata = hacide.test)
pred_over_rpart <- predict(over_rpart, newdata = hacide.test)
pred_under_rpart <- predict(under_rpart, newdata = hacide.test)
pred_both_rpart <- predict(both_rpart, newdata = hacide.test)
pred_rose_rpart <- predict(rose_rpart, newdata = hacide.test)

roc.curve(hacide.test$cls, pred_raw_rpart[, 2], plot = TRUE)
roc.curve(hacide.test$cls, pred_over_rpart[, 2], plot = TRUE)
roc.curve(hacide.test$cls, pred_under_rpart[, 2], plot = TRUE)
roc.curve(hacide.test$cls, pred_both_rpart[, 2], plot = TRUE)
roc.curve(hacide.test$cls, pred_rose_rpart[, 2], plot = TRUE)

raw_roc_df <- tibble(cls = hacide.test[, 1], pred = pred_raw_rpart[, 2],
                     sampling = "Raw")
over_roc_df <- tibble(cls = hacide.test[, 1], pred = pred_over_rpart[, 2],
                      sampling = "Over")
under_roc_df <- tibble(cls = hacide.test[, 1], pred = pred_under_rpart[, 2],
                       sampling = "Under")
both_roc_df <- tibble(cls = hacide.test[, 1], pred = pred_both_rpart[, 2],
                       sampling = "Both")
rose_roc_df <- tibble(cls = hacide.test[, 1], pred = pred_rose_rpart[, 2],
                      sampling = "ROSE")

hacide_roc_df <- bind_rows(list(raw_roc_df, over_roc_df,
                                under_roc_df, both_roc_df,
                                rose_roc_df))

ggplot(hacide_roc_df, aes(d = cls, m = pred, color = sampling)) +
    geom_roc(labels = FALSE) +
    style_roc() +
    theme_pubr() +
    theme(legend.position = "top")
