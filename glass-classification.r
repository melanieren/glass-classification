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

# Bind the scaled input data with the Type column and verify the result
scaled_data <- cbind(scaled_data_input, Type=data$Type)
head(scaled_data)

# Create train and test sets
set.seed(30)

index <- sample.int(n = nrow(scaled_data), size = floor(0.75*nrow(scaled_data)), replace = F)

train <- as.data.frame(scaled_data[index, ])
test <- as.data.frame(scaled_data[-index, ])

head(train)

## Create some predictive models
# General linear fit
lm_fit <- glm(Type~., data=train)
summary(lm_fit)

# Compare the linear model predictions with the actual Type 
lm_predict <- predict(lm_fit, test)
lm_predictions <- cbind(test, PredictedType=round(lm_predict))

# Calculate the RMSE
lm_RMSE <- sqrt(sum((lm_predict - test$Type)^2)/nrow(test))
print(lm_RMSE)

# The RMSE for the linear model is 1.172145, which will be the benchmark for now



