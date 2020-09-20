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
#train_total_f <- train_total_f[-c(1)]
#test_total_f <- test_total_f[-c(1)]

library(dplyr)
train_total_f <- train_total_f %>% mutate_if(is.character,as.factor)
test_total_f <- test_total_f %>% mutate_if(is.character,as.factor)

#Fitting a Random Forest Classifier
library(caret)
library(randomForest)
bid <- test_total$building_id
dataset <- train_total_f
dataset_test <- test_total_f
dataset$damage_grade <- as.factor(dataset$damage_grade)
library(caTools)
split = sample.split(dataset$damage_grade, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
colnames(training_set)

classifier = randomForest(x = training_set[,-9],
                          y = training_set$damage_grade , ntree = 15)  

plot(classifier)
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