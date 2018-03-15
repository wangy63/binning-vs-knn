# Load bin-stacked ensembles package (install if needed from my GitHub)

# library(devtools)
# devtools::install_github("kmaurer/binsemble")
library(binsemble)

# Iris example
names(iris)[5] <- "true_class"
set.seed(12345)
train_index <- c(sample(1:50, 30),sample(51:100, 30),sample(101:150, 30))
train <- iris[train_index, ]
test <- iris[-train_index, ]
true_classes <- levels(iris$true_class)

# -------
## Specify member classifiers with function
model_types <- c("weka.classifiers.bayes.NaiveBayes","weka.classifiers.trees.RandomForest",
                 "weka.classifiers.meta.Bagging","weka.classifiers.functions.SMO")

model_list <- make_model_list(model_types, test)
model_list[[1]]

# -------
## Make them test ensembles

train_preds <- make_train_preds(train_data=train,model_list=model_list,true_classes=levels(iris$true_class))

# simple weighted ensemble
weightedEnsemble <- make_ensemble(train_preds=train_preds, model_list=model_list, weightType="weighted", comb_rule="majority vote")
predictEnsemble(weightedEnsemble, test)
eval_ensemble(weightedEnsemble, test)

# bin weighted ensemble
weightedEnsemble <- make_ensemble(train_preds=train_preds, model_list=model_list, weightType="bin weighted", comb_rule="average posterior",
                                  bin_type="standard", bin_features=c("'Petal.Length','Petal.Width'"), nbins=c(2,2))
predictEnsemble(weightedEnsemble, test)
eval_ensemble(weightedEnsemble, test)

# knn weighted ensemble
weightedEnsemble <- make_ensemble(train_preds=train_preds, model_list=model_list, weightType="knn", comb_rule="majority vote", knn_size=20)
predictEnsemble(weightedEnsemble, test)
eval_ensemble(weightedEnsemble, test)
