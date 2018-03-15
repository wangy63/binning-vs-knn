library(MASS)
library(tidyverse)
library(randomForest)
library(e1071)
library(data.table)
library(ggplot2)
library(FNN)
set.seed(12345)
setwd("/Users/wangyuexi/Desktop/research/Dr. Maurer PJ/Data for SP")
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
scalesubset<-scale(subset[, 1:7])
train_scale<-data.frame(cbind(scalesubset[train_index, ]), subset[train_index, 8])
names(train_scale)[8]<-"true_class"
test_scale<-data.frame(cbind(scalesubset[-train_index, ]), subset[-train_index, 8])
names(test_scale)[8]<-"true_class"

#### maybe change to 10-fold CV ####
## add other model such as QDA, KNN: Algorithm="kd_tree", "_tree"
##### LOOCV on LDA ####
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
train.scale_cv <- add_cv_cohorts(train_scale,10)
head(train.scale_cv)
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


lda_rf_knn_pred <- matrix(ncol=3, nrow=nrow(train))
for (cv_k in 1:10){
  print(cv_k)
  route=0
  cohort_rows <- which(train_cv$cv_cohort == cv_k)
  youtube_lda <- lda(true_class ~ ., data = train[-cohort_rows , ])
  youtube_rf<-randomForest(true_class~. , data = train[-cohort_rows , ], mtry=1)
  knnClass <- knn(train = train_scale[-cohort_rows , -8],
                  test = train_scale[cohort_rows , -8],
                  cl = train_scale[-cohort_rows , "true_class"], k = 10, algorithm = "kd_tree")
  for(i in cohort_rows){
    lda_rf_knn_pred[i, 1]<-predict(youtube_lda, train[i , ])[[1]]
    lda_rf_knn_pred[i, 2]<-as.numeric(predict(youtube_rf, train[i , ]))
    route<-route+1
    lda_rf_knn_pred[i, 3] <- as.numeric(knnClass)[route]
  }
}
compare<-cbind(as.data.frame(lda_rf_knn_pred), as.numeric(train$true_class))
names(compare)<-c("lda_pred", "rf_pred", "knn_pred", "true")



## find the distance
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
find_10NN<-proc.time()-prev

#### sort the 10 NN for each test data
prev<-proc.time()
size=100
NNfull<-data.frame(lda_pred=NA,  rf_pred=NA, knn_pred=NA, true=NA, testpoint=NA)
for (i in 1:nrow(distance)){
  print(i)
  NN_index<-order(distance[i,])[1:size]
  NN_sub<-data.frame(lda_pred=NA, rf_pred=NA, knn_pred=NA, true=NA)
  for (m in c(NN_index)){
    NN_predict<-compare[m, ]
    NN_sub<- rbind(NN_sub, NN_predict)
  }
  NN_sub$testpoint<-i
  NNfull<-rbind(NN_sub, NNfull)
}
order<-proc.time()-prev

#!## check ####
NNfull<-na.omit(NNfull)
NNfull$lda_pred=ifelse(NNfull$lda_pred==NNfull$true,1,0)
NNfull$rf_pred=ifelse(NNfull$rf_pred==NNfull$true,1,0)
NNfull$knn_pred=ifelse(NNfull$knn_pred==NNfull$true,1,0)


#### find accuracy for the neighbors for two different models for test data ####
NNaccuracy<-NNfull%>%
  group_by(testpoint)%>%
  summarize(accuracy_lda=sum(lda_pred)/size, accuracy_rf=sum(rf_pred)/size, accuracy_knn=sum(knn_pred)/size)

#### choose which model the test point go with #####
choose_model<-proc.time()-prev
test_pred<-data.frame(testpoint=NA, model=NA, prediction=NA)
youtube_lda <- lda(true_class ~ ., data = train)
youtube_rf<-randomForest(true_class~. , data = train, mtry=1)
knnClass <- knn(train = train_scale[-cohort_rows , -8],
                test = train_scale[cohort_rows , -8],
                cl = train_scale[-cohort_rows , "true_class"], k = 10, algorithm = "kd_tree")


for (i in 1:nrow(NNaccuracy)){
  print(i)
  test_pred[i, 1]<-i
  if (NNaccuracy$accuracy_lda[i]==max(NNaccuracy[i, 2:4])){
    test_pred$model[i]<-"lda"
    test_pred$prediction[i]<-predict(youtube_lda,test[i, ])[[1]]
  } else if (NNaccuracy$accuracy_rf[i]==max(NNaccuracy[i, 2:4])){
    test_pred$model[i]<-"rf"
    test_pred$prediction[i]<-predict(youtube_rf, test[i, ])
  } 
  
}
  
  
# else (NNaccuracy$accuracy_knn[i]==max(NNaccuracy[1, 2:4])){
#    test_pred$model[i]<-"knn"
#    knnClass <- knn(train = train_scale[ , -8],
#                    test = test_scale[i , -8],
#                    cl = train_scale[ , "true_class"], k = 10, algorithm = "kd_tree")
#    as.numeric(knnClass)
#    test_pred$prediction[i]<-predict(youtube_knn, test[i, ])
#  }
#}




##### calculate the overall accuracy #####
test_pred<-cbind(test_pred, test$true_class)
names(test_pred)[4]<-"true_class"
test_pred$true_class<-as.numeric(test_pred$true_class)
test_pred$correctness<-ifelse(test_pred$prediction==test_pred$true_class,1,0)
sum(test_pred$correctness)/nrow(test_pred)
## 0.6193497
save(NNaccuracy, file="NNaccuracy")



plot_data<-cbind(test, test_pred$model)
names(plot_data)[9]<-"model"

ggplot(data=plot_data, aes(x=width, y=duration, color=model))+
  geom_jitter()+
  ggtitle("k=100 model selection")
