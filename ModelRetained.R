library(dplyr)
library(ggplot2)
library(tidyr)
#read data
pathWD = "C:\\Users\\Himel\\OneDrive\\Studium\\M.Sc. Statistics\\MA\\ML_Challenge"
dataRaw = read.csv(file.path(pathWD, "data.csv"), sep = ",", 
                   stringsAsFactors = TRUE)
dataRaw = dataRaw %>% select(-c(id, last_name))

#summary
summ_data = summarizeColumns(dataRaw)
m[as.vector(sapply(m, is.numeric))] = round(m[as.vector(sapply(m, is.numeric))], 3)
m


#Box plot for quantitative variables
dataRaw$retained = as.factor(ifelse(dataRaw$retained == "1", "yes", "no"))
#dataRaw$retained = as.factor(dataRaw$retained)

gg.violin = function(dataset, variable){
  ggplot(dataRaw, aes_string(x=retained, y= variable)) + 
    geom_violin(trim=FALSE, fill="gray") + labs(x="retained", y = variable)+
    geom_boxplot(width=0.1)+
    theme_classic()
} 

Var_quant = c("credit_score", "age", "years_customer", 
              "balance_euros", "num_products", "salary_euros")

VioPlotAll = lapply(FUN = gg.violin, X = Var_quant, dataset = dataRaw)
ggarrange(VioPlotAll[[1]], VioPlotAll[[2]],
          VioPlotAll[[3]],VioPlotAll[[4]],
          VioPlotAll[[5]], VioPlotAll[[6]],nrow = 2, ncol = 3)


#Bar plot for categorial variable
gg.bar = function(dataset, variable){
  ggplot(data = dataset) + 
    geom_bar(mapping = aes_string(x = variable, fill = "retained"))+
    theme_classic() + scale_fill_brewer(palette="Blues")
}

var_categorial = c("country", "gender", "has_credit_card", 
                   "is_active", "retained")

BarPlotAll = lapply(FUN = gg.bar, X = var_categorial, dataset = dataRaw)
ggarrange(BarPlotAll[[1]], BarPlotAll[[2]],
          BarPlotAll[[3]],BarPlotAll[[4]],
          BarPlotAll[[5]], nrow = 2, ncol = 3, common.legend = TRUE)


#correlation plot for quantitative variable
cor = cor(dataRaw[var_quant])
corrplot(cor, type = "full", order = "hclust", 
         tl.col = "black", tl.srt = 45) 





#--------------------RF Model-----------------------------------#

#model
library("caret")
set.seed(123)
idx.train <- caret::createDataPartition(y = dataRaw$retained, p = 0.75, list = FALSE) # Draw a random, stratified sample including p percent of the data
tr <- dataRaw[idx.train, ] # training set
ts <-  dataRaw[-idx.train, ] # test set (drop all observations with train indeces)

task <- makeClassifTask(data = tr, target = "retained" , positive = "yes")
task

rf <- makeLearner("classif.randomForest", 
                  predict.type = "prob", # prediction type needs to be specified for the learner 
                  par.vals = list("replace" = TRUE, "importance" = FALSE,"nodesize"=5))
rf


## Tuning
# Hyperparameter setting
# Set the scale/range for your parameters
rf.parms <- makeParamSet(
  # The recommendation for mtry by Breiman is squareroot number of columns
  makeIntegerParam("mtry", lower = 2, upper = 6), # Number of features selected at each node, smaller -> faster
  makeDiscreteParam("sampsize", values = c(200,300)), # bootstrap sample size, smaller -> faster
  makeIntegerParam("ntree", lower = 300, upper = 1000) # Number of tree, smaller -> faster
) 


# How dense should the parameters be selected from the ranges?
tuneControl <- makeTuneControlGrid(resolution = 3, tune.threshold = FALSE)

# Sampling strategy
# Given our small dataset, we do cross-validation
rdesc <- makeResampleDesc(method = "CV", iters = 10, stratify = TRUE)

# Start tuning with the defined options
tuning <- tuneParams(rf, task = task, resampling = rdesc,
                     par.set = rf.parms, control = tuneControl, measures = mlr::auc)



# Update the learner to the optimal hyperparameters
rf
rf_tuned <- setHyperPars(rf, par.vals = tuning$x)
rf_tuned

# Train the model on the full training data (not only a CV-fold)
model_library <- list()
#yhat <- list()

model_library[["rf"]] <- mlr::train(rf_tuned, task = task)

# Make prediction on test data
#yhat[["rf"]] <- predict(modelLib[["rf"]], newdata = ts)
#str(yhat[["rf"]])

# Calculate AUC performance on test set 
#mlr::performance(yhat[["rf"]], measures = mlr::auc)




#---------------------------------------------------------------------------------



# Prepare the mlr task
# Xgboost doesn't take categorical variables as input
data_dummy <- mlr::createDummyFeatures(dataRaw, target="retained")
train_data <- data_dummy[idx.train, ] # training set
test_data <-  data_dummy[-idx.train, ] # test set (drop all observations with train indeces)

task <- makeClassifTask(data = train_data, target = "retained", positive = "yes")

library("xgboost")
xgb.learner <- makeLearner("classif.xgboost", predict.type = "prob", # prediction type needs to be specified for the learner
                           par.vals = list("verbose" = 0,
                                           "early_stopping_rounds"=20)) # early stopping when no improvement for k iterations
xgb.learner

# Set tuning parameters
xgb.parms <- makeParamSet(
  makeNumericParam("eta", lower = 0.01, upper = 0.05), 
  makeIntegerParam("nrounds", lower=80, upper=400), 
  makeIntegerParam("max_depth", lower=2, upper=6),
  makeDiscreteParam("gamma", values = 0),
  makeDiscreteParam("colsample_bytree", values = 1),
  makeDiscreteParam("min_child_weight", values = 1),
  makeDiscreteParam("subsample", values = 1)
)

# How dense should the parameters be selected from the ranges?
tuneControl <- makeTuneControlRandom(maxit=100, tune.threshold = FALSE)

# We do 3-fold cross-validation, given the small data more folds might be better
rdesc <- makeResampleDesc(method = "RepCV", rep = 3, folds=2, stratify = TRUE)


set.seed(123) # Set seed for the local random number generator, e.g. the CV samples
# Tune parameters as before
xgb.tuning <- tuneParams(xgb.learner, task = task, resampling = rdesc,
                         par.set = xgb.parms, control = tuneControl, measures = mlr::auc)


# Extract optimal parameter values after tuning 
xgb.tuning$x

# Update the learner to the optimal hyperparameters
xgb.learner <- setHyperPars(xgb.learner, par.vals = c(xgb.tuning$x, "verbose" = 0))
xgb.learner

# Train the model on the full training data (not only a CV-fold)
#model_library <- list()
model_library[["xgb"]] <- mlr::train(xgb.learner, task = task)


#--------------------------------------------------------------------------------------


#logistic regression
#dataRaw$retained = ifelse(dataRaw$retained == "yes", 1, 0)
data_dummy <- mlr::createDummyFeatures(dataRaw, target="retained")
train_data <- data_dummy[idx.train, ] # training set
test_data <-  data_dummy[-idx.train, ] # test set (drop all observations with train indeces)



# Also train gradient boosting with a better set of hyperparameters found by extensive grid search
#xgb.learner <- setHyperPars(xgb.learner, par.vals = list("eta"=0.03,"nrounds"=300, "max_depth"=4, "verbose" = 0))
#model_library[["xgb"]] <- mlr::train(xgb.learner, task = task)

# Make prediction on test data
pred <- sapply(model_library, predict, newdata = test_data, simplify=FALSE)
# Calculate AUC performance on test set 
auc <- sapply(pred, mlr::performance, measures = mlr::auc)
# Compare the gradient boosting performance to last week's random forest
auc


#boxplot of Auc
boxplot(tuning_results$data$auc.test.mean, main = "Boxplot: AUC of parameter tuning")

#Roc curve
pred.roc = data.frame("RF" = pred$rf$data$prob.yes, "XGb" = pred$xgb$data$prob.yes)
h = HMeasure(true.class = as.numeric(test_data$retained == "yes"), scores = pred.roc)
plotROC(h, which = 1)
