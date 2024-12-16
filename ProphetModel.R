library(tidymodels)
library(vroom)
library(modeltime)
library(timetk)
library(prophet)

train <- vroom('train.csv')
test <- vroom('test.csv')


nStores <- max(train$store)
nItems <- max(train$item)

for(s in 1:nStores){
  for(i in 1:nItems){
    storeitemtrain <- train |>
      filter(store == s, item == i)
    storeitemtest <- test |>
      filter(store == s, item == i)
    
    prophet_model <- prophet_reg() |>
      set_engine(engine = 'prophet') |>
      fit(sales ~ date, data = storeitemtrain)
    
    cv_results_fb <- modeltime_calibrate(prophet_model, 
                                         new_data = testing(cv_split))
    fullfit <- cv_results_fb |>
      modeltime_refit(data = train_1)
    
    preds <- fullfit %>%
      modeltime_forecast(
        new_data = storeItemTest,
        actual_data = storeItemTrain
      ) %>%
      filter(!is.na(.model_id)) %>%
      mutate(id=storeItemTest$id) %>%
      select(id, .value) %>%
      rename(sales=.value)
    
    if(s == 1 & i == 1){
      all_preds <- preds
    } else {
      all_preds <- bindrows(all_preds, preds)
    }
  }
}

vroom_write(all_preds, file = 'submission.csv', delim = ',')

