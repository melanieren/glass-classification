# Read in the data
library(readr)
data <- read.csv("data/glass.csv")

# Install packages
install.packages("caret")
library(caret)

# Overview of the dataset
head(data)
summary(data)

## Exploratory data analysis
# Univariate
# Histogram
par(mfrow=c(1,9))
for(i in 1:9) {
	hist(data[,i], main=names(data)[i])
}

# Boxplot 
par(mfrow=c(1,9))
for(i in 1:9) {
	boxplot(data[,i], main=names(data)[i])
}

# Multivariate
# Correlation plot 
install.packages("corrplot")
library(corrplot)
correlations <- cor(data[,1:9])
corrplot(correlations, method="circle")

# Scatterplot matrix by class
pairs(Type~.,data=data,col=data$Type)

# Density by class
x <- data[,1:9]
y <- data[,10]
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

# Boxplots by class
featurePlot(x=x, y=y, plot="box")

## Prepare the data for modelling
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
fit_lda <- train(Type~., data=train, method="lda", metric=metric, trControl=control)

# CART
set.seed(5)
fit_cart <- train(Type~., data=train, method="rpart", metric=metric, trControl=control)

# kNN
set.seed(5)
fit_knn <- train(Type~., data=train, method="knn", metric=metric, trControl=control)

# SVM
set.seed(5)
fit_svm <- train(Type~., data=train, method="svmRadial", metric=metric, trControl=control)

# Random Forest
set.seed(5)
fit_rf <- train(Type~., data=train, method="rf", metric=metric, trControl=control)

# Neural Net
set.seed(5)
fit_nnet <- train(Type~., data=train, method="nnet", metric=metric, trControl=control)

# Compare the results of the models 
results <- resamples(list(lda=fit_lda, cart=fit_cart, knn=fit_knn, svm=fit_svm, rf=fit_rf, nnet=fit_nnet))

summary(results)
dotplot(results)

# The best model based on accuracy 
print(fit_rf)

predictions <- predict(fit_rf, test)
confusionMatrix(predictions, test$Type)

