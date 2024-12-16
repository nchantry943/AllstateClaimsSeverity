library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(rpart)
library(ranger)
library(dbarts)
library(embed)

train <- vroom('train.csv')
test <- vroom('test.csv')

bart_mod <- parsnip::bart(trees = 100) |>
  set_engine('dbarts') |>
  set_mode('regression') |>
  translate()

my_recipe <- recipe(loss ~ ., data = train) %>%
  step_rm(id) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.6) %>% 
  step_normalize(all_numeric_predictors())%>% 
  step_zv(all_predictors())

prep_rec <- prep(my_recipe)
new_train <- bake(prep_rec, new_data = train)

workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(bart_mod) |>
  fit(data = train)

lin_pred <- predict(workflow, new_data = test)

workflow_pred <- lin_pred %>%
  mutate(id = test$id) %>%
  mutate(loss = .pred) %>% 
  select(2, 3)

vroom_write(x=workflow_pred, file = "./BARTPred.csv", delim=",")


