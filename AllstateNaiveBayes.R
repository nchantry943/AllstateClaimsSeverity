library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(rpart)
library(ranger)
library(dbarts)

train <- vroom('train.csv')
test <- vroom('test.csv')

bart_mod <- parsnip::bart(trees = 100) |>
  set_engine('dbarts') |>
  set_mode('regression') |>
  translate()

my_recipe <- recipe(loss ~ ., data = train) |>
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) |>
  step_impute_knn(all_of(cat_columns), impute_with = imp_vars(loss), neighbors = 5)

workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(bart_mod) |>
  fit(data = train)

lin_pred <- predict(workflow, new_data = test)

workflow_pred <- lin_pred |>
  bind_cols(test) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=workflow_pred, file = "./BARTPred.csv", delim=",")


