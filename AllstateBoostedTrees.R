library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(bonsai)
library(lightgbm)

## Read in Data
train <- vroom('train.csv')
test <- vroom('test.csv')


## Receipe
my_recipe <- recipe(loss ~ ., data = train) |>
  step_rm(id) |> 
  step_other(all_nominal_predictors(), threshold = .001) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) |> 
  step_corr(all_numeric_predictors(), threshold = 0.6) |> 
  step_normalize(all_numeric_predictors())|> 
  step_zv(all_predictors())



model <- boost_tree(
  mode = "regression",
  engine = "lightgbm",
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune()
)


## Workflow
workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(model)


grid <- grid_regular(trees(),
                     tree_depth(),
                     learn_rate(),
                     levels = 3)

fold <- vfold_cv(train, v = 5, repeats = 1)

CV <- workflow |>
  tune_grid(resamples = fold, grid = grid, metrics = metric_set(rmse, mae, rsq))

best <- CV |> select_best(metric = 'mae')
best

final_wf <- workflow |>
  finalize_workflow(best) |>
  fit(data = train)

pred <- predict(final_wf, new_data = test)

workflow_pred <- pred |>
  mutate(id = test$id) |>
  mutate(loss = .pred) |> 
  select(2, 3)

vroom_write(x=workflow_pred, file = "./BoostedTreesPred3.csv", delim=",")
save(file = '/workflow_pred.RData', list = c('workflow_pred'))