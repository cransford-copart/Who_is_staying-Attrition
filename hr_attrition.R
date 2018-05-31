#################### Kaggle - Employee Attrition  - rf, gbm, and xgboost ###################################################
library(ggplot2)
library(tidyverse)
library(RcppRoll)
library(caret)
library(randomForest)
library(Metrics)
library(gbm)

empl_att <- read.csv("C:/Users/cjransford/Documents/R Scripts/Kaggle/WA_Fn-UseC_-HR-Employee-Attrition.csv")

empl_att$Education <- as.factor(empl_att$Education)
empl_att$StockOptionLevel <- as.factor(empl_att$StockOptionLevel)
empl_att$StandardHours <- as.integer(empl_att$StandardHours)


# Remove StandardHours and EmployeeNumber and EmployeeCount - as each row in these fields have the same value
empl_att <- empl_att[, -which(names(empl_att) %in% c("StandardHours","EmployeeNumber","EmployeeCount","Over18"))]

apply(is.na(empl_att),2,sum) # No NA's found in the data set



hist(empl_att$ï..Age)

ggplot(empl_att, aes(x = ï..Age, fill = Attrition)) + geom_histogram(breaks = seq(20, 60, 2))
ggplot(empl_att, aes(x = MonthlyIncome, fill = Attrition)) + geom_histogram()
ggplot(empl_att, aes(x = YearsAtCompany, fill = Attrition)) + geom_histogram(breaks = seq(0, 40, 2))



# Separate training and validation test sets
in_train_att <- createDataPartition(empl_att$Attrition, p = 0.7, list = FALSE)

train_att <- empl_att[in_train_att,]
validation_att <- empl_att[-in_train_att,]



# Setup tuning for GBM hyperparameters
# Ratio of TRUE to FALSE is ~ 1:5, which is still enough to create a bias in the data - conduct oversampling to reduce bias
gbmGrid <- expand.grid(interaction.depth = c(1,3,6,9,12),
                       n.trees = seq(50,1000,50),#,
                       shrinkage = 0.01,
                       n.minobsinnode = seq(5,30,5))

fit_control <- trainControl(method = "repeatedcv",
                            number = 5,                        # relatively small data set - use fewer cross validations
                            repeats = 3,
                            sampling = "up",                   # perform upsampling (oversampling)
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary)



set.seed(212)
gbm_att <- train(Attrition ~ ., 
                 data = train_att,
                 distribution = "bernoulli",
                 method = "gbm", 
                 # bag.fraction = 0.5,
                 # nTrain = round(nrow(train_att) *.75),
                 trControl = fit_control,
                 verbose = FALSE,
                 tuneGrid = gbmGrid,
                 metric = "ROC")                         # Since test set will have bias in the response variable, auc is the more appropriate measure


plot(gbm_att)
print(varImp(gbm_att))



# Assess performance for both models on test data

# Confusion Matrix
confusionMatrix(predict(gbm_att, validation_att),
                validation_att$Attrition)

# AUC
gbmPreds <- predict(gbm_att, validation_att)

pROC::roc(as.numeric(validation_att$Attrition), as.numeric(gbmPreds))














# Xgboost model
install.packages("vtreat")
library(vtreat)
library(magrittr)
library(xgboost)

# CReate treatment plan for categorical variables - later used on both training and test sets
treatVars <- c(
  "ï..Age", "Attrition", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome", "Education",  "EducationField", "EnvironmentSatisfaction", "Gender",
  "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",  "MonthlyIncome",  "MonthlyRate", "NumCompaniesWorked", "OverTime", 
  "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
  "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion"
)


treatPlan <- designTreatmentsZ(train_att, treatVars, verbose = FALSE)

scoreFrame <- treatPlan %>%
  magrittr::use_series(scoreFrame) %>%
  select(varName, origName, code)

# We only want the rows with codes "clean" or "lev"
newvars <- scoreFrame %>%
  filter(code %in% c("clean","lev")) %>%
  use_series(varName)

train_att.treat <- prepare(treatPlan, train_att, varRestriction = newvars)
validation_att.treat <- prepare(treatPlan, validation_att, varRestriction = newvars)

train_att.treat_dep <- train_att.treat[,c(1:22,25:52)]
train_att.treat_resp <- train_att.treat[,c(24)]
dtrain <- xgb.DMatrix(data = train_att.treat_dep, label = )


xgb_params = list(
  objective = "binary:logistic",                                               # binary classification
  eta = 0.01,                                                                  # learning rate
  max.depth = 10,                                                              # max tree depth
  eval_metric = "auc"                                                          # evaluation/loss metric
)


xgbAtt <- xgb.cv(data = as.matrix(train_att.treat_dep),
                 label = train_att.treat_resp,
                 nrounds = 500,
                 nfold = 5,
                 # showsd = TRUE,
                 # stratified = TRUE,
                 # maximize = FALSE,
                 params = xgb_params,
                 early_stopping_rounds = 20,
                 verbose = 0
)

# Get the evaluation log 
elog <- as.data.frame(xgbAtt$evaluation_log)

# Determine and print how many trees minimize training and test error
elog %>% 
  summarize(ntrees.train = which.max(train_auc_mean),   # find the index of min(train_rmse_mean)
            ntrees.test  = which.max(test_auc_mean))





xgbControl <- trainControl(method = "cv",
                           number = 5,# relatively small data set - use fewer cross validations
                           sampling = "up",                   # perform upsampling (oversampling)
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)


xgbGrid <- expand.grid(nrounds = seq(50,500,25),
                       eta = 0.01,
                       gamma = .01,
                       max_depth = c(4,6,8,10),
                       subsample = 0.7,
                       colsample_bytree = 0.7,
                       min_child_weight = 1)
names(train_att)
set.seed(211)
xgb2 <- train(Attrition ~ ., 
              data = train_att[,c(1:10,12:17,19:31)],
              method = "xgbTree",
              trControl = xgbControl,
              tuneGrid = xgbGrid,
              verbose = 0,
              metric = "ROC",
              maximize = TRUE)


varImp(xgb2)
plot(xgb2)

xgb2$bestTune
xgb2$finalModel


confusionMatrix(predict(xgb2, newdata = validation_att[,c(1:10,12:17,19:31)]),
                validation_att$Attrition)

# AUC
xgbPreds <- predict(xgb2, validation_att)

pROC::plot.roc(as.numeric(validation_att$Attrition), as.numeric(xgbPreds), print.auc = TRUE)

pROC::plot.roc(as.numeric(validation_att$Attrition), as.numeric(xgbPreds), print.auc = TRUE, print.thres = .5, type = "S")













# find the index of min(test_rmse_mean)


xgb1 <- xgboost(data = as.matrix(train_att.treat_dep),
                label = train_att.treat_resp,
                params = xgb_params,
                nrounds = 83,
                nfold = 5,
                # showsd = TRUE,
                # stratified = TRUE,
                # maximize = FALSE,
                early_stopping_rounds = 20,
                verbose = 0)



confusionMatrix(predict(xgb1, newdata = validation_att.treat),
                validation_att.treat$)











fitControl <- trainControl(method="cv", 
                           number = 5, 
                           classProbs = TRUE,
                           sampling = "up")

xgbGrid <- expand.grid(nrounds = 200,
                       max_depth = 12,
                       eta = .03,
                       gamma = 0.01,
                       colsample_bytree = .7,
                       min_child_weight = 1,
                       subsample = 0.9)



xgbControl <- trainControl(method = "cv",
                           number = 5,                        # relatively small data set - use fewer cross validations
                           sampling = "up",                   # perform upsampling (oversampling)
                           classProbs = TRUE)
#summaryFunction = twoClassSummary)


xgbGrid <- expand.grid(nrounds = seq(50,500,25),
                       eta = .03,
                       gamma = .01,
                       max_depth = c(4,6,8,10),
                       subsample = 0.9,
                       colsample_bytree = 0.7,
                       min_child_weight = 1)


XGB.model <- train(formula, 
                   data = validation_att,
                   method = "xgbTree",
                   trControl = xgbControl,
                   verbose=0,
                   maximize=FALSE,
                   tuneGrid = xgbGrid,
                   eval_metric = "AUC"
)