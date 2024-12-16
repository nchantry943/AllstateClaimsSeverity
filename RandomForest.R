library(tidymodels)
library(forecast)
library(vroom)
library(ggplot2)
library(rpart)
library(embed)
library(bonsai)
library(lightgbm)

test <- vroom('test.csv')
train <- vroom('train.csv')

my_recipe <- recipe(loss ~ ., data = train) %>%
  step_rm(id) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.6) %>% 
  step_normalize(all_numeric_predictors())%>% 
  step_zv(all_predictors())
  
 

rand_for_mod <- rand_forest(mtry = tune(),
                            min_n = tune(), 
                            trees = 500) |>
  set_engine('ranger') |>
  set_mode("regression")

rand_work <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(rand_for_mod)

grid1 <- grid_regular(mtry(range = c(2, round(sqrt(ncol(train))))),
                      min_n(range = c(2, 20)),
                      levels = 4)

fold1 <- vfold_cv(train, v = 5, repeats = 1)

CV1 <- rand_work |>
  tune_grid(resamples = fold1, 
            grid = grid1, 
            metrics = metric_set(rmse, mae, rsq))

best1 <- CV1 |> select_best(metric = 'mae')

final_wf1 <- rand_work |>
  finalize_workflow(best1) |>
  fit(data = train)

for_pred <- predict(final_wf1, new_data = test)

rand_pred <- for_pred %>%
  mutate(id = test$id) %>%
  mutate(loss = .pred) %>% 
  select(2, 3)

vroom_write(x=rand_pred, file = "./RandomForestPred.csv", delim=",")
