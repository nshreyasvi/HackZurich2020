rm(list=ls())
train_data <- read.csv("C:/Users/shrey/Downloads/public_dat/public_data/train.csv", sep=',', header = T)
building_structure <- read.csv("C:/Users/shrey/Downloads/public_dat/public_data/building_structure.csv", sep=',', header = T)
building_ownership <- read.csv("C:/Users/shrey/Downloads/public_dat/public_data/building_ownership.csv", sep=',', header = T)
ward_demographic <- read.csv("C:/Users/shrey/Downloads/public_dat/public_data/ward_demographic_data.csv", sep=',', header = T)
test_data <- read.csv("C:/Users/shrey/Downloads/public_dat/public_data/test.csv", sep=',', header = T)

#=====================================================================================================
#Merging the dataset into one giant dataset for training set and testing set
order(train_data$building_id)
order(test_data$building_id)
order(building_ownership$building_id)
order(building_structure$building_id)

train_total <- merge(train_data,building_ownership,by=c("building_id"))
train_total_f <- merge(train_total, building_structure, by=c("building_id"))

test_total <- merge(test_data, building_ownership, by=c("building_id"))
test_total_f <- merge(test_total, building_structure, by=c("building_id"))

#Removing the first column i.e. building ID as it is unrelated
train_total_f <- train_total_f[-c(1)]
test_total_f <- test_total_f[-c(1)]

#Converting each of the labels into right format i.e. categorical variables and numerical variables
summary(train_total_f)

library(dplyr)
train_total_f <- train_total_f %>% mutate_if(is.character,as.factor)
test_total_f <- test_total_f %>% mutate_if(is.character,as.factor)
levels(train_total_f$ground_floor_type)
levels(train_total_f$foundation_type)
#====================================================================================================
#Running Tests for just Numerical Values (Ignoring non-numerical values)
#Finding Correlation
#Chose this if using only non-character categorical variables
#correlation_train_set <- train_total_f[,-which(sapply(train_total_f, class) == "character")]
#correlation_test_set <- test_total_f[,-which(sapply(test_total_f, class) == "character")]

#normal set
#correlation_train_set <- train_total_f
#correlation_test_set <- test_total_f

#Choose this if using the entire dataset
correlation_train_set <- train_total_f %>% mutate_if(is.factor,as.integer)
correlation_test_set <- test_total_f %>% mutate_if(is.factor,as.integer)

#correlation_train_set$plinth_area_sq_ft <- log(correlation_train_set$plinth_area_sq_ft)
#correlation_train_set$height_ft_pre_eq <- log(correlation_train_set$height_ft_pre_eq)
#correlation_train_set$age_building <- sqrt(correlation_train_set$age_building)
#correlation_test_set$plinth_area_sq_ft <- log(correlation_test_set$plinth_area_sq_ft)
#correlation_test_set$height_ft_pre_eq <- log(correlation_test_set$height_ft_pre_eq)
#correlation_test_set$age_building <- sqrt(correlation_test_set$age_building)

#correlation_train_set$district_id.x <- log(correlation_train_set$district_id.x)
#correlation_train_set$district_id.y <- log(correlation_train_set$district_id.y)
#correlation_train_set$vdcmun_id.x <- log(correlation_train_set$vdcmun_id.x)
#correlation_train_set$vdcmun_id.y <- log(correlation_train_set$vdcmun_id.y)
#correlation_train_set$ward_id.x <- log(correlation_train_set$ward_id.x)
#correlation_train_set$ward_id.y <- log(correlation_train_set$ward_id.y)
#correlation_test_set$district_id.x <- log(correlation_test_set$district_id.x)
#correlation_test_set$district_id.y <- log(correlation_test_set$district_id.y)
#correlation_test_set$vdcmun_id.x <- log(correlation_test_set$vdcmun_id.x)
#correlation_test_set$vdcmun_id.y <- log(correlation_test_set$vdcmun_id.y)
#correlation_test_set$ward_id.x <- log(correlation_test_set$ward_id.x)
#correlation_test_set$ward_id.y <- log(correlation_test_set$ward_id.y)

colnames(correlation_test_set)

correlation_train_set <- correlation_train_set[-c(10,11,12,14)]
correlation_test_set <- correlation_test_set[-c(9,10,11,13)]
#correlation_train_set <- correlation_train_set[-c(10,11,12,14,26,27,28)]
#correlation_test_set <- correlation_test_set[-c(9,10,11,13,25,26,27)]

summary(correlation_train_set)

#====================================================================================================
library(corrplot)

test <- cor(correlation_train_set, correlation_train_set$damage_grade)
#corrplot(test, method="number")
test
write.table(test,file="correlation.csv")
#Finding Significant correlation between some variables

cor.test(correlation_train_set$has_secondary_use_gov_office, correlation_train_set$damage_grade)
cor.test(correlation_train_set$has_superstructure_mud_mortar_brick, correlation_train_set$damage_grade)

#========================================================================================================
#Creating a decision tree for the 5 classes
library(rpart)
library(ggplot2)
library(rpart.plot)
# tree based classification
fit <- rpart(damage_grade ~ .,
             method="class", data=correlation_train_set)
# plot tree
rpart.plot(fit, type =5, extra = 101, digits=-3)
summary(fit)

library(ggfortify)
#correlation_train_set<-na.omit(correlation_train_set)
#correlation_train_set<-na.omit(correlation_test_set)

pca_set <- prcomp(correlation_train_set,scale. = TRUE)
summary(pca_set)

#=====================================================================================================
#Gradient booster method
library(gbm)
set.seed(123)
gbm_fit <- gbm(
  formula = damage_grade ~ .,
  distribution = "gaussian",
  data = correlation_train_set,
  n.trees = 15,
  interaction.depth = 5,
  shrinkage = 0.1,
  n.minobsinnode = 5,
  bag.fraction = .65, 
  train.fraction = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)

summary(gbm_fit, 
        cBars = 15,
        method = relative.influence, # also can use permutation.test.gbm
        las = 1
)
#====================================================================================================
#Fitting a Random Forest Classifier
library(caret)
library(randomForest)
dataset <- correlation_train_set
dataset_test <- correlation_test_set
dataset$damage_grade <- as.factor(dataset$damage_grade)
#dataset <- na.omit(dataset)
library(caTools)
split = sample.split(dataset$damage_grade, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
colnames(training_set)

classifier = randomForest(x = training_set[,-9],
                          y = training_set$damage_grade , ntree = 15)  

#plot(classifier)
y_pred = predict(classifier, newdata = test_set)
#y_pred = predict(classifier, newdata = dataset_test)
y_pred

building_id <- test_total$building_id
#Use this to write the output variable/dataframe into a csv file for further usage
#length(y_pred)
#write.csv(data.frame(building_id, y=y_pred), file='prediction.csv', row.names=FALSE)

# Making the Confusion Matrix
cm = table(test_set[,9], y_pred)
cm
print("====================================Random Forest=====================================")
library(ggplot2)
library(lattice)
library(caret)
confusionMatrix(cm)
#For Full Dataset: Balanced Accuracy     0.78263  0.67038  0.65114   0.6811   0.7749 (For Class 1-5)
#FOr Semi Dataset: Balanced Accuracy     0.77837  0.66751  0.64551   0.6761   0.7699 (For Class 1-5)
#===================================================================================================
#Ride Lasso and ElasticNet Implementation
library(glmnet)

split = sample.split(dataset$damage_grade, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#change alpha=1 (ridge), alpha=0 (lasso), alpha=0.5 (elasticnet)
ridge.fit <- glmnet(x=as.matrix(training_set[,-9]),y=training_set[,9],
                    family='multinomial',alpha=0.5)

plot(ridge.fit,xvar='lambda',label=TRUE)

nlam<-length(ridge.fit$lambda)
ridge.pred.tr<-predict(ridge.fit,newx=as.matrix(training_set[,-9]),
                       type = 'class')
ridge.pred.te<-predict(ridge.fit,newx=as.matrix(test_set[,-9]),
                       type = 'class')

y_pred = predict(ridge.fit, newx =  as.matrix(dataset_test),type = 'class')
y_pred
write.csv(data.frame(bid, y=y_pred), file='prediction.csv', row.names=FALSE)


ridge.train <- ridge.test <- numeric(nlam)
for (i in 1:nlam){
  ridge.train[i] <- mean(!(ridge.pred.tr[,i]==training_set$damage_grade))
  ridge.test[i] <- mean(!(ridge.pred.te[,i]==test_set$damage_grade))
}
#To check for output accuracy with the current model
plot(log(ridge.fit$lambda),ridge.train,type='l')
lines(log(ridge.fit$lambda),ridge.test,col='red')
lines(log(ridge.fit$lambda),rep(0,nlam),lty='dotdash')

#===================================================================================================================
library(caTools)
library(lattice)
library(caret)
library(C50)
#Trimmed dataset 
colnames(correlation_train_set)
#correlation_train_set <- correlation_train_set[-c(6,7,14,19,20,21,22,23,24,25,34,38)]
colnames(correlation_train_set)
correlation_train_set$damage_grade <- as.factor(correlation_train_set$damage_grade)

split = sample.split(correlation_train_set$damage_grade, SplitRatio = 0.75)
training_set = subset(correlation_train_set, split == TRUE)
test_set = subset(correlation_train_set, split == FALSE)

fit <- C5.0(training_set[-9], training_set$damage_grade, trials = 5)
saveRDS(fit, file = "c5.0.rds")
summary(fit)
#plot(fit)

fit
dataset_test <- correlation_test_set
y_pred = predict(fit, newdata = dataset_test)
damage_grade <-y_pred

#Use this to write the output variable/dataframe into a csv file for further usage
building_id <- test_total$building_id
write.csv(data.frame(building_id, y=damage_grade), file='prediction.csv', row.names=FALSE)
