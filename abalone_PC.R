abalone<-cbind(abalone$V1,as.data.frame(scale(abalone[, 2:9])))
dat<-abalone
mypca <- principal(cor(abalone[,-1]), nfactors=8)
mypca$values

#### binsemble ####
model_types <- c("weka.classifiers.bayes.NaiveBayes","weka.classifiers.trees.RandomForest",
                 "weka.classifiers.meta.Bagging","weka.classifiers.functions.SMO")
nbins_list<-list(c(2, 2), c(3, 3), c(5, 5), c(8, 2), c(12, 2),c(12, 4), c(18, 2))


pc_accuracy<-data.frame(accuracy=NA, weightType=NA, comb_rule=NA, bin_type=NA, nbins=NA, seed=NA)
bin_feature_list <- list(levels(top_index)[top_index],levels(top_index)[mid_index],levels(top_index)[low_index])
counter <- 0
prev<-Sys.time()
for (seed in 1:50){
  print(seed)
  set.seed(seed)
  train_index <- sample(seq_len(nrow(dat)), size = 0.75*nrow(dat))
  train <- dat[train_index, ]
  test <- dat[-train_index, ]
  model_list <- make_model_list(model_types, train)
  train_preds <- make_train_preds(train_data=train,model_list=model_list,true_classes=levels(dat$true_class))
  for (k in c("bin weighted")){
    for (j in c("average posterior", "majority vote")){
      for (m in c("standard", "quantile", "iterative quantile")){
        print(m)
        for (n in 1:7){
          counter <- counter+1
          weightedEnsemble <- make_ensemble(train_preds=train_preds, model_list=model_list, weightType=k, comb_rule=j,
                                            bin_type=m, bin_features=c("PC1", "PC2"), nbins= nbins_list[[n]], rotate=TRUE)
          pc_accuracy[counter,1]<-eval_ensemble(weightedEnsemble, test)[1,2]
          pc_accuracy[counter,2]<-k
          pc_accuracy[counter,3]<-j
          pc_accuracy[counter,4]<-m
          pc_accuracy[counter, 5]<-nbins_list[[n]][1]*nbins_list[[n]][2]
          pc_accuracy[counter,6]<-seed
          
        }
      }
    }
  }
}
time<-Sys.time()-prev

pcdata<-pc_accuracy%>%
  group_by(comb_rule, bin_type, nbins)%>%
  summarise(accuracy=mean(accuracy))
pcdata$bin_feature<-"PC"

pcdata$bin_feature<-as.character(pcdata$bin_feature)
pcdata$nbins<-as.factor(pcdata$nbins)
plotdata_abalone<-rbind(normal_abalone, pcdata)
plotdata_abalone$nbins<-as.numeric(plotdata_abalone$nbins)

dat<-plotdata_abalone[order(plotdata_abalone$nbins), ]
dat$nbins<-as.factor(dat$nbins)

pcplot_full<-ggplot(data=dat,aes(x=nbins, y=accuracy, color=bin_feature, group=bin_feature))+
  facet_grid(comb_rule ~ bin_type)+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("summarize of data")
abalone_PC<-dat
save(abalone_PC, file="abalone_PC")
setwd("/Users/wangyuexi/Desktop/research/Dr. Maurer PJ/datasets")
load("abalone_PC")
load("market_PC")
load("satimage_PC")
