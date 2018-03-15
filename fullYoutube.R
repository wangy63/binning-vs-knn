library(MASS)
library(tidyverse)
library(randomForest)
library(e1071)
library(data.table)
library(ggplot2)
library(FNN)
library(dplyr)
set.seed(12345)
setwd("/Users/wangy63/Downloads")
youtube<-read.csv("youtube_videos.csv")
youtube<-youtube[, c(-1, -9, -11)]
youtube<-as.data.frame(youtube)
names(youtube)[8] <- "true_class"
youtube$true_class<-as.factor(youtube$true_class)
true_classes <- levels(youtube$true_class)
subset_index <- sample(seq_len(nrow(youtube)), size = 0.01*nrow(youtube))
subset <- youtube[subset_index, ]
train_index<-sample(seq_len(nrow(subset)), size = 0.75*nrow(subset))
train<-subset[train_index, ]
test<-subset[train_index, ]


#### maybe change to 10-fold CV ####
##### LOOCV on LDA & RF ####
add_cv_cohorts <- function(dat,cv_K){
  if(nrow(dat) %% cv_K == 0){ # if perfectly divisible
    dat$cv_cohort <- sample(rep(1:cv_K, each=(nrow(dat)%/%cv_K)))
  } else { # if not perfectly divisible
    dat$cv_cohort <- sample(c(rep(1:(nrow(dat) %% cv_K), each=(nrow(dat)%/%cv_K + 1)),
                              rep((nrow(dat) %% cv_K + 1):cv_K,each=(nrow(dat)%/%cv_K)) ) )
  }
  return(dat)
}

# add 10-fold CV labels to real estate data
train_cv <- add_cv_cohorts(train,10)
head(train_cv)

## Use the 10 groups to iteratively fit and check the 5-nearest neighbors model
# initialize for 10 cohorts, errors and counts
cohorts <- data.frame(cohort=1:10,
                      errors = rep(NA,10), 
                      n=rep(NA,10))

#### 10-fold CV on lda ####
misclass_rf <- rep(NA,10)
for(m in 1:10){
  print(m)
  mod <- randomForest(true_class~. , data=train, mtry=m)
  # OOB test misclassification rate is average rate of mis
  misclass_rf[m] <- sum(mod$predicted != youtube$true_class)/nrow(youtube)
}
misclass_rf

prev<-Sys.time()
lda_rf_bagging_pred <- matrix(ncol=3, nrow=nrow(train))
for (cv_k in 1:10){
  print(cv_k)
  cohort_rows <- which(train_cv$cv_cohort == cv_k)
  youtube_lda <- lda(true_class ~ ., data = train[-cohort_rows , ])
  youtube_bagging <- randomForest(true_class~. , data = train[-cohort_rows , ], mtry=7)
  youtube_rf<-randomForest(true_class~. , data = train[-cohort_rows , ], mtry=1)
  for(i in cohort_rows){
    lda_rf_bagging_pred[i, 1]<-predict(youtube_lda, train[i , ])[[1]]
    lda_rf_bagging_pred[i, 2]<-as.numeric(predict(youtube_bagging, train[i , ]))
    lda_rf_bagging_pred[i, 3]<-as.numeric(predict(youtube_rf, train[i , ]))
  }
}
CV<-Sys.time()-prev
compare<-cbind(as.data.frame(lda_rf_bagging_pred), as.numeric(train$true_class))
names(compare)<-c("lda_pred", "bagging_pred","rf_pred", "true")



## find the distance
prev<-Sys.time()
distance<-matrix(NA, nrow=nrow(test), ncol=nrow(test))
for (i in 1:nrow(test)){
  print(i)
  for (j in 1:nrow(train)){
    sum<-0
    for (k in 1: 7){    ##because the first column is "true class"
      sum<-(test[i, k]-train[j, k])^2+sum
    }
    distance[i, j]<-sqrt(sum)
  }
}
find.distance<-Sys.time()-prev

#### sort the 10 NN for each test data
prev<-proc.time()
size=100
NNfull<-data.frame(lda_pred=NA, bagging_pred=NA, rf_pred=NA, true=NA, testpoint=NA)
for (i in 1:nrow(distance)){
  print(i)
  NN_index<-order(distance[i,])[1:size]
  NN_sub<-data.frame(lda_pred=NA,bagging_pred=NA, rf_pred=NA, true=NA)
  for (m in c(NN_index)){
    NN_predict<-compare[m, ]
    NN_sub<- rbind(NN_sub, NN_predict)
  }
  NN_sub$testpoint<-i
  NNfull<-rbind(NN_sub, NNfull)
}
order<-proc.time()-prev

NNfull<-na.omit(NNfull)
NNfull$lda_pred=ifelse(NNfull$lda_pred==NNfull$true,1,0)
NNfull$bagging_pred=ifelse(NNfull$bagging_pred==NNfull$true,1,0)
NNfull$rf_pred=ifelse(NNfull$rf_pred==NNfull$true,1,0)


#### find accuracy for the neighbors for two different models for test data ####
NNaccuracy<-NNfull%>%
  group_by(testpoint)%>%
  summarize(accuracy_lda=sum(lda_pred)/size, accuracy_bagging=sum(bagging_pred)/size, accuracy_rf=sum(rf_pred)/size)

#### choose which model the test point go with #####
choose_model<-proc.time()-prev
test_pred<-data.frame(testpoint=NA, model=NA, prediction=NA)
youtube_lda <- lda(true_class ~ ., data = train)
youtube_bagging<-randomForest(true_class~. , data = train, mtry=7)
youtube_rf<-randomForest(true_class~. , data = train, mtry=1)

prev<-Sys.time()
for (i in 1:nrow(NNaccuracy)){
  print(i)
  test_pred[i, 1]<-i
  if (NNaccuracy$accuracy_lda[i]==max(NNaccuracy[i, 2:4])){
    test_pred$model[i]<-"lda"
    test_pred$prediction[i]<-predict(youtube_lda,test[i, ])[[1]]
  } 
  else if (NNaccuracy$accuracy_rf[i]==max(NNaccuracy[i, 2:4])){
    test_pred$model[i]<-"rf"
    test_pred$prediction[i]<-predict(youtube_rf, test[i, ])
  } 
  else {
    test_pred$model[i]<-"bagging"
    test_pred$prediction[i]<-predict(youtube_bagging,test[i, ])
  }
}
find.model<-Sys.time()-prev


##### calculate the overall accuracy #####
test_pred<-cbind(test_pred, test$true_class)
names(test_pred)[4]<-"true_class"
test_pred$true_class<-as.numeric(test_pred$true_class)
test_pred$correctness<-ifelse(test_pred$prediction==test_pred$true_class,1,0)
final.accuracy<-sum(test_pred$correctness)/nrow(test_pred)

save(NNaccuracy, file="NNaccuracy.Rdata")

plot_data<-cbind(test, test_pred$model)
names(plot_data)[9]<-"model"

plot_3model<-ggplot(data=plot_data, aes(x=width, y=duration, color=model))+
  geom_jitter()+
  ggtitle("k=100 model selection")
save(plot_3model, file="plot_3model.data")
ggplotly(plot)
