# Read in the data
library(readr)
data <- read.csv("data/glass.csv")

# Install packages
install.packages("caret")
library(caret)

# Overview of the dataset
head(data)
summary(data)

# Scale the column values, except for the Type column (output column)
to_drop <- c('Type')
data_input <-  data[ , !(names(data) %in% to_drop)]
head(data_input)
scaled_data_input <- scale(data_input)

# Convert Type column to a factor so caret will recognize this as a classification problem 
data$Type <- as.factor(data$Type)
levels(data$Type) <- c("building_float", "building_non_float", "vehicle_float", "vehicle_non_float", "containers", "tableware")

# Bind the scaled input data with the Type column and verify the result
scaled_data <- cbind(scaled_data_input, Type=data$Type)
head(scaled_data)

# Create train and test sets
set.seed(30)

index <- sample.int(n = nrow(scaled_data), size = floor(0.75*nrow(scaled_data)), replace = F)

train <- as.data.frame(scaled_data[index, ])
test <- as.data.frame(scaled_data[-index, ])

train$Type <- as.factor(train$Type)
levels(train$Type) <- c("building_float", "building_non_float", "vehicle_float", "vehicle_non_float", "containers", "tableware")
test$Type <- as.factor(test$Type)
levels(test$Type) <- c("building_float", "building_non_float", "vehicle_float", "vehicle_non_float", "containers", "tableware")

head(train)

# Use cross-validation to avoid overfitting 
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

## Create some predictive models
# LDA
set.seed(5)
fit.lda <- train(Type~., data=train, method="lda", metric=metric, trControl=control)

# CART
set.seed(5)
fit.cart <- train(Type~., data=train, method="rpart", metric=metric, trControl=control)

# kNN
set.seed(5)
fit.knn <- train(Type~., data=train, method="knn", metric=metric, trControl=control)

# SVM
set.seed(5)
fit.svm <- train(Type~., data=train, method="svmRadial", metric=metric, trControl=control)

# Random Forest
set.seed(5)
fit.rf <- train(Type~., data=train, method="rf", metric=metric, trControl=control)

results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

dotplot(results)

print(fit.rf)

predictions <- predict(fit.rf, test)
confusionMatrix(predictions, test$Type)