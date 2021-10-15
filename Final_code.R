
library(ISLR)
library(glmnet)
library(dplyr)
library(tidyr)
library(ggplot2)
library(tree)
library(rpart)          
library(rpart.plot)     
library(randomForest)   
library(gbm)   
library(class)
require(randomForest)
library(ggplot2)
library(dplyr)
library(LICORS)
library(ggiraphExtra)
library(neuralnet)



set.seed(1)
setwd("C:/Users/alex/Desktop/STATISTICS COURSES/STA 141A/Final Project")


train.bank <- read.table("bank-additional/bank-additional.csv", head = T, sep = ";")

full.bank <- read.table("bank-additional/bank-additional-full.csv", head = T, sep = ";")
head(train.bank)

###Training Data Manipulation

set.seed(1)
data.train = train.bank
data.train[data.train == "unknown"] = NA
str(data.train)

data.train$job = data.train$job %>% as.factor() %>% as.numeric()
data.train$marital = data.train$marital %>% as.factor() %>% as.numeric()
data.train$education = data.train$education %>% as.factor() %>% as.numeric() 
data.train$default = data.train$default %>% as.factor() %>% as.numeric()
data.train$housing = data.train$housing %>% as.factor() %>% as.numeric() 
data.train$loan = data.train$loan %>% as.factor() %>% as.numeric()
data.train$contact = data.train$contact %>% as.factor() %>% as.numeric() 
data.train$month = data.train$month %>% as.factor() %>% as.numeric() 
data.train$day_of_week = data.train$day_of_week %>% as.factor() %>% as.numeric()
data.train$poutcome = data.train$poutcome %>% as.factor() %>% as.numeric()




x_training = model.matrix(y~.,data.train)[,-1]
head(x_training)




y_training = data.train$y %>% 
  unlist() %>%
  as.factor() %>%
  as.numeric()

y_training = (ifelse(y_training == 2, 1, 0))
y_training
one = which(y_training == 0)
one = length(one)
two = length(data.train$y)

one/two



###Full Data Manipulation

data.full = full.bank
data.full[data.full == "unknown"] = NA
data.full = na.omit(data.full)

data.full$job = data.full$job %>% as.numeric()
data.full$marital = data.full$marital %>% as.numeric()
data.full$education = data.full$education %>% as.numeric() 
data.full$default = data.full$default %>% as.numeric()
data.full$housing = data.full$housing %>% as.numeric() 
data.full$loan = data.full$loan %>% as.numeric()
data.full$contact = data.full$contact %>% as.numeric() 
data.full$month = data.full$month %>% as.numeric() 
data.full$day_of_week = data.full$day_of_week %>% as.numeric()
data.full$poutcome = data.full$poutcome %>% as.numeric()

x_full = model.matrix(y~.,data.full)[,-1]

y_full = data.full$y %>% 
  unlist() %>%
  as.numeric()

y_full = (ifelse(y_full == 2, 1, 0))

##### Model Selection


lasso_mod = glmnet(x_training,y_training,alpha = 1,family = "binomial")
plot(lasso_mod)


cv.out = cv.glmnet(x_training,y_training,alpha = 1,family = "binomial")

plot(cv.out)
bestlam = cv.out$lambda.min
bestlam

out = glmnet(x_full, y_full, alpha = 1) 
lasso_coef = predict(out, type = "coefficients", s = bestlam)[1:21,]
lasso_coef


rem_feat = c(5,6,14)


### Logistic Regression and Probabilities

library(pROC)
logistic_model = glm(y~.,data = data.train[,-rem_feat],family = "binomial")

probabilities = logistic_model %>% predict(data.full[,-rem_feat], type = "response")
probabilities
head(probabilities)
classes = ifelse(probabilities > 0.5, 1, 0)
head(classes)
mean(classes == y_full)


length(y_full)
length(classes)

test_prob1 = predict(logistic_model, newdata = data.full, type = "response")
roc_fit1 = roc(data.full$y ~ test_prob1, plot = TRUE, print.auc = TRUE)

as.numeric(test_roc$auc)


#### Tree Classification
set.seed(1)
training_tree_data = train.bank
training_tree_data[training_tree_data == "unknown"] = NA
training_tree_data = na.omit(training_tree_data)

data.full.tree = full.bank
data.full.tree[data.full.tree == "unknown"] = NA
data.full.tree = na.omit(data.full.tree)

fit = rpart(y~.,data = training_tree_data[,-rem_feat],method = "class")
printcp(fit)
plotcp(fit)

tree.cv = rpart(y~.,data = training_tree_data[,-rem_feat],
                method = "class", cp = 0.013514  )
rpart.plot(tree.cv)


(treeT <- table(predict(tree.cv, data.full.tree, type = "class"),data.full.tree$y))
table(predict(tree.cv, training_tree_data, type = "class"),training_tree_data$y)
(treeT[1,1]+treeT[2,2])/(sum(treeT))


test_prob2 = predict(tree.cv, newdata = data.full.tree, type = "prob")
test_prob
str(data.full.tree)

#data.full.tree$y = ifelse(data.full.tree$y == "yes",1,0)
roc_test2 = roc(data.full.tree$y ~ test_prob2[,2], plot = TRUE, print.auc = TRUE)
as.numeric(test_roc$auc)



#### Random Forest

head(training_tree_data)
rf.random = randomForest(y~., data = training_tree_data,importance = T)
rf.random$confusion
importance(rf.random)
varImpPlot(rf.random)

prob_fit = predict(rf.random,newdata = data.full.tree, type = "prob")
head(prob_fit)
roc_fit3 =roc(data.full.tree$y ~ prob_fit[,1],plot = TRUE, print.auc = TRUE)




### ROC PLOTS




plot(roc_fit1,col = "Red")
plot(roc_test2,add = TRUE,col = "Blue")
plot(roc_fit3,add = TRUE, col = "Green")









### EXtra Credit Neural Networks
library(tidyverse)
library(neuralnet)
set.seed(1)



data.train = train.bank
data.train[data.train == "unknown"] = NA
data.train$job = data.train$job %>% as.numeric()
data.train$marital = data.train$marital %>% as.numeric()
data.train$education = data.train$education %>% as.numeric() 
data.train$default = data.train$default %>% as.numeric()
data.train$housing = data.train$housing %>% as.numeric() 
data.train$loan = data.train$loan %>% as.numeric()
data.train$contact = data.train$contact %>% as.numeric() 
data.train$month = data.train$month %>% as.numeric() 
data.train$day_of_week = data.train$day_of_week %>% as.numeric()
data.train$poutcome = data.train$poutcome %>% as.numeric()

data.train = na.omit(data.train)


data.train$y = ifelse(data.train$y == "yes",1,0)


head(data.train)


data.full = full.bank
data.full[data.full == "unknown"] = NA
data.full = na.omit(data.full)

data.full$job = data.full$job %>% as.numeric()
data.full$marital = data.full$marital %>% as.numeric()
data.full$education = data.full$education %>% as.numeric() 
data.full$default = data.full$default %>% as.numeric()
data.full$housing = data.full$housing %>% as.numeric() 
data.full$loan = data.full$loan %>% as.numeric()
data.full$contact = data.full$contact %>% as.numeric() 
data.full$month = data.full$month %>% as.numeric() 
data.full$day_of_week = data.full$day_of_week %>% as.numeric()
data.full$poutcome = data.full$poutcome %>% as.numeric()

data.full$y = ifelse(data.full$y == "yes",1,0)

head(data.full)



nn = neuralnet(y~., data = data.train[,-rem_feat],linear.output = FALSE, hidden = 3,err.fct = "sse", act.fct = 
                 "logistic")
plot(nn)

Predict = predict(nn,data.full[,-rem_feat], type = "response")


prob <- Predict
pred <- ifelse(prob>0.5, 1, 0)

head(y_full)
mean(pred == y_full)










