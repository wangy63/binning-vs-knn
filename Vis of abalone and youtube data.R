library(ggplot2)
library(dplyr)
library(gridExtra)
################## iterative quantile ###############
### size of the data and time it spend ####
## setup a dataset for size and average speed. 
time_size<-data.table(matrix(NA, nrow=4, ncol=3))
time_size[, 1]<-c("youtube", "rock", "credit", "youtube")
names(time_size)<-c("dataset", "size", "avg_time")
time_size$size<-c(nrow(abalone), nrow(Rock), nrow(credit), nrow(youtube))
time_size$avg_time<-c(mean(result_abalone$time), mean(result_rock$time), mean(result_credit$time), mean(result_youtube$time))
time_size_graph<-ggplot(data=time_size, aes(x=size, y=avg_time))+
  geom_smooth()+
  geom_point()+
  ggtitle("time spend for differnet size of data (in second)")+
  xlab("size of data")+
  ylab("average time spend on computing IQbin")

# as the size getting larger, the time spend would growth in an exponnential way
####### accuracy vs nbins ######
#### abalone ###
accuracy_abalone_iq<-result_abalone_iq %>% 
  group_by(nbins_name) %>% 
  summarise(max_accuracy=max(accuracy), avg_accuracy=mean(accuracy), variance=var(rep(accuracy)))

max_nbin_accuracy_abalone_iq<-ggplot(data=accuracy_abalone, aes(x=nbins_name, y=max_accuracy, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("max accuracy for different nbins")+
  xlab("nbins (X*Y)")

avg_nbin_accuracy_abalone_iq<-ggplot(data=accuracy_abalone, aes(x=nbins_name, y=avg_accuracy, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("avg accuracy for different nbins")+
  xlab("nbins (X*Y)")

#### credit ###
max_accuracy_credit<-result_credit %>% 
  group_by(nbins_name) %>% 
  top_n(1, accuracy)

max_nbin_accuracy_credit<-ggplot(data=max_accuracy_credit, aes(x=nbins_name, y=accuracy, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("accuracy for different nbins")+
  xlab("nbins (X*Y)")


####youtube ###
accuracy_youtube_iq<-result_youtube_iq %>% 
  group_by(nbins_name) %>% 
  summarise(max_accuracy=max(accuracy), avg_accuracy=mean(accuracy), variance=var(rep(accuracy)))

max_nbin_accuracy_youtube_iq<-ggplot(data=accuracy_youtube_iq, aes(x=nbins_name, y=max_accuracy, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("max accuracy for different nbins")+
  xlab("nbins (X*Y)")

avg_nbin_accuracy_youtube_iq<-ggplot(data=accuracy_youtube_iq, aes(x=nbins_name, y=avg_accuracy, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("avg accuracy for different nbins")+
  xlab("nbins (X*Y)")

##### variance of accuracy with each nbins for different bin variable#######
## abalone
var_acc_abalone_plot_iq<-ggplot(data=accuracy_abalone_iq, aes(x=nbins_name, y=variance, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("variance of accuracy within each nbins")+
  xlab("nbins (X*Y")
grid.arrange(var_acc_abalone_plot_iq, max_nbin_accuracy_abalone_iq, avg_nbin_accuracy_abalone_iq, ncol=2)
# note: nbins with lower variance also has lower accuracy, higher variance also has highe accuracy, which is conradict to what I expect

## youtube data
var_acc_youtube_plot<-ggplot(data=accuracy_youtube, aes(x=nbins_name, y=variance, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("variance of accuracy within each nbins")+
  xlab("nbins (X*Y")
grid.arrange(var_acc_youtube_plot, max_nbin_accuracy_youtube, avg_nbin_accuracy_youtube, ncol=2)


###### variable used to bin vs accuracy (mean and variance)
## abalone
diffvar_abalone<-result_abalone %>% 
  group_by(bin_pair_name) %>% 
  summarise(max_accuracy=max(accuracy), avg_accuracy=mean(accuracy), variance=var(rep(accuracy)))

diffvar_abalone_max<-ggplot(data=diffvar_abalone, aes(x=bin_pair_name, y=max_accuracy, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("max accuracy for different paired bins")+
  xlab("nbins (X*Y)")

diffvar_abalone_avg<-ggplot(data=diffvar_abalone, aes(x=bin_pair_name, y=avg_accuracy, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("avg accuracy for different paired bins")+
  xlab("nbins (X*Y)")

diffvar_abalone_variance<-ggplot(data=diffvar_abalone, aes(x=bin_pair_name, y=variance, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("variance of accuracy for different paired bins")+
  xlab("nbins (X*Y)")

grid.arrange(diffvar_abalone_max, diffvar_abalone_avg, diffvar_abalone_variance, ncol=2)

# variance and accuracy would increase and decrease at the same time

## youtube data
diffvar_youtube<-result_youtube %>% 
  group_by(nbins_name) %>% 
  summarise(max_accuracy=max(accuracy), avg_accuracy=mean(accuracy), variance=var(rep(accuracy)))

diffvar_youtube_max<-ggplot(data=diffvar_youtube, aes(x=nbins_name, y=max_accuracy, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("max accuracy for different paired bins")+
  xlab("nbins (X*Y)")

diffvar_youtube_avg<-ggplot(data=diffvar_youtube, aes(x=nbins_name, y=avg_accuracy, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("avg accuracy for different paired bins")+
  xlab("nbins (X*Y)")

diffvar_youtube_variance<-ggplot(data=diffvar_youtube, aes(x=nbins_name, y=variance, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("variance of accuracy for different paired bins")+
  xlab("nbins (X*Y)")

grid.arrange(diffvar_youtube_max, diffvar_youtube_avg, diffvar_youtube_variance, ncol=2)

## maybe for a large size data, the variance would growth in opposite direction of average accuracy (shown in the above plot1 and plot2)

################### standard binnig ###########
#### abalone ###
accuracy_abalone_Standard<-result_abalone_standard %>% 
  group_by(nbins_name) %>% 
  summarise(max_accuracy=max(accuracy), avg_accuracy=mean(accuracy), variance=var(rep(accuracy)))

max_nbin_accuracy_abalone_stan<-ggplot(data=accuracy_abalone_Standard, aes(x=nbins_name, y=max_accuracy, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("max accuracy for different nbins")+
  xlab("nbins (X*Y)")

avg_nbin_accuracy_abalone_stan<-ggplot(data=accuracy_abalone_Standard, aes(x=nbins_name, y=avg_accuracy, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("avg accuracy for different nbins")+
  xlab("nbins (X*Y)")

var_acc_abalone_plot_stan<-ggplot(data=accuracy_abalone_Standard, aes(x=nbins_name, y=variance, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("variance of accuracy within each nbins")+
  xlab("nbins (X*Y")
grid.arrange(var_acc_abalone_plot_stan, max_nbin_accuracy_abalone_stan, avg_nbin_accuracy_abalone_stan, ncol=2)


#### youtube ##
accuracy_youtube_Standard<-result_youtube_standard %>% 
  group_by(nbins_name) %>% 
  summarise(max_accuracy=max(accuracy), avg_accuracy=mean(accuracy), variance=var(rep(accuracy)))

max_nbin_accuracy_youtube_stan<-ggplot(data=accuracy_youtube_Standard, aes(x=nbins_name, y=max_accuracy, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("max accuracy for different nbins")+
  xlab("nbins (X*Y)")

avg_nbin_accuracy_youtube_stan<-ggplot(data=accuracy_youtube_Standard, aes(x=nbins_name, y=avg_accuracy, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("avg accuracy for different nbins")+
  xlab("nbins (X*Y)")

var_acc_youtube_plot_stan<-ggplot(data=accuracy_youtube_Standard, aes(x=nbins_name, y=variance, group=1))+
  geom_line(linetype = "dashed")+
  geom_point()+
  ggtitle("variance of accuracy within each nbins")+
  xlab("nbins (X*Y")
grid.arrange(var_acc_youtube_plot_stan, max_nbin_accuracy_youtube_stan, avg_nbin_accuracy_youtube_stan, ncol=2)

########### combine for abalone ######
accuracy_abalone_iq$group<-"iq"
accuracy_abalone_Standard$group<-"standard"
comb_data_abalone<-rbind(accuracy_abalone_Standard, accuracy_abalone_iq)
save(comb_data_abalone, file="comb_abalone.Rdata")

var_nbins_comb<-ggplot(data=comb_data_abalone, aes(x=nbins_name, y=variance, color=group))+
  geom_line(linetype = "dashed", aes(group = group))+
  geom_point()+
  ggtitle("variance of accuracy within each nbins")+
  xlab("nbins (X*Y)")

max_acc_nbins_comb<-ggplot(data=comb_data_abalone, aes(x=nbins_name, y=max_accuracy, color=group))+
  geom_line(linetype = "dashed", aes(group = group))+
  geom_point()+
  ggtitle("max accuracy within each nbins")+
  xlab("nbins (X*Y)")+
  geom_hline(aes(yintercept = 0.5358852))+
  geom_text(aes(y=0.5358852, label="unweighted", x=5), colour="blue", angle=0, vjust = 1.2, text=element_text(size=11))+
  geom_hline(aes(yintercept = 0.5406699))+
  geom_text(aes(y=0.5406699, label="weighted", x=5), colour="purple", angle=0, vjust = 1.2, text=element_text(size=11))
               
avg_acc_nbins_comb<-ggplot(data=comb_data_abalone, aes(x=nbins_name, y=avg_accuracy, color=group))+
  geom_line(linetype = "dashed", aes(group = group))+
  geom_point()+
  ggtitle("max accuracy within each nbins")+
  xlab("nbins (X*Y)")+
  geom_hline(aes(yintercept = 0.5358852))+
  geom_text(aes(y=0.5358852, label="unweighted", x=5), colour="blue", angle=0, vjust = 1.2, text=element_text(size=11))+
  geom_hline(aes(yintercept = 0.5406699))+
  geom_text(aes(y=0.5406699, label="weighted", x=5), colour="purple", angle=0, vjust = 1.2, text=element_text(size=11))


grid.arrange(var_nbins_comb, max_acc_nbins_comb, avg_acc_nbins_comb, ncol=2)

######## youtube data #######
########### combine for abalone ######
accuracy_youtube_iq$group<-"iq"
accuracy_youtube_Standard$group<-"standard"
comb_data_youtube<-rbind(accuracy_youtube_Standard, accuracy_youtube_iq)
save(comb_data_youtube, file="comb_youtube.Rdata")

var_nbins_comb_youtube<-ggplot(data=comb_data_youtube, aes(x=nbins_name, y=variance, color=group))+
  geom_line(linetype = "dashed", aes(group = group))+
  geom_point()+
  ggtitle("variance of accuracy within each nbins")+
  xlab("nbins (X*Y)")

max_acc_nbins_comb_youtube<-ggplot(data=comb_data_youtube, aes(x=nbins_name, y=max_accuracy, color=group))+
  geom_line(linetype = "dashed", aes(group = group))+
  geom_point()+
  ggtitle("max accuracy within each nbins")+
  xlab("nbins (X*Y)")+
  geom_hline(aes(yintercept = 0.248))+
  geom_text(aes(y=0.248, label="unweighted", x=5), colour="blue", angle=0, vjust = 1.2, text=element_text(size=11))+
  geom_hline(aes(yintercept = 0.228))+
  geom_text(aes(y=0.228, label="weighted", x=5), colour="purple", angle=0, vjust = 1.2, text=element_text(size=11))




avg_acc_nbins_comb_youtube<-ggplot(data=comb_data_youtube, aes(x=nbins_name, y=avg_accuracy, color=group))+
  geom_line(linetype = "dashed", aes(group = group))+
  geom_point()+
  ggtitle("max accuracy within each nbins")+
  xlab("nbins (X*Y)")+
  geom_hline(aes(yintercept = 0.248))+
  geom_text(aes(y=0.248, label="unweighted", x=5), colour="blue", angle=0, vjust = 1.2, text=element_text(size=11))+
  geom_hline(aes(yintercept = 0.228))+
  geom_text(aes(y=0.228, label="weighted", x=5), colour="purple", angle=0, vjust = 1.2, text=element_text(size=11))


grid.arrange(var_nbins_comb_youtube, max_acc_nbins_comb_youtube, avg_acc_nbins_comb_youtube, ncol=2)


### wednesday at 9:30