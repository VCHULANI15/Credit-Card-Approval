library(magrittr)
library(dplyr)
library(stringr)
library(ggplot2)
library(caret)
library(klaR)
library(reshape2)
library(corrplot)
setwd("/Users/apple/Desktop/Data Science Final Project")
# import data
application <- read.csv("application.data.csv")

# Separate train and test data. 75% train, 25% test
set.seed(129)
index <- sample(nrow(application), nrow(application) * 0.75)
application_train <- application[index,]
application_test <- application[-index,]


####################################################################################################
#################################    Data Visualization    #########################################
####################################################################################################


############# 1. Visualization Analysis Related to Total Income  ############
#############################################################################

summary(application$Total_Income)
# bar chart of income
par(plt=c(0.09, 0.8, 0.18, 0.8))
par(cex.main = 1.2)
par(mai=c(0.8, 1, 1, 0.9))
hist(application$Total_Income, breaks = 100, xlab = 'Total Income', ylab = 'Frequency', main = 'Total Income Distribution')

# Calculate the proportion of applicants with income greater than 500,000    1.32%
proportion_income <- sum(application$Total_Income > 500000) / nrow(application)
proportion_income




### 2. Clustering Income and Bad Debt Across Different Job Positions and Box Plot Analysis  ###
###############################################################################################

# create subsets for Laborers. Then conduct K-means. Last draw box plot
application$Job_Title<-gsub(" ","",application$Job_Title)
application$Job_Title<-factor(application$Job_Title)
Laborers<-subset(application,Job_Title=='Laborers')

Kmeans_data <- subset(Laborers,select=c("Total_Income","Total_Bad_Debt"))
S_Kmeans_data <- scale(Kmeans_data)
kmeans_results<-kmeans(S_Kmeans_data,4, nstart=10)
kmeans_results
colorcluster <- 1+kmeans_results$cluster
plot(S_Kmeans_data, col = 1)
plot(S_Kmeans_data, col = colorcluster, xlab = 'Total Income', ylab = 'Total Bad Debt', main = 'Scatter Plot with K-means Clustering(Laborers)')
points(kmeans_results$centers, col = 1, pch = 24, cex = 1.5, lwd=1, bg = 2:5)

cluster_data <- split(S_Kmeans_data, kmeans_results$cluster)
cluster_values <- lapply(cluster_data, unlist)
boxplot(cluster_values, main = "Cluster Boxplot of Laborers", xlab = "Cluster", ylab = "Total Bad Debt Value")



# income boxplot by job title
unique_jobs <- unique(application$Job_Title)
unique_jobs
library(tidyverse)

p <- ggplot(application, aes(x = Job_Title, y = Total_Income))
p + geom_boxplot() +
  labs(title = "Income Boxplot by Job Title",
       x = "Job Title",
       y = "Total Income") +
  theme(axis.text.x = element_text(angle = 45, hjust = 0.5))



# approval rate within different occupations
df1 <- application %>%
  group_by(Job_Title) %>%
  summarize(avgApproval = mean(Status)) # IT Staff lowest
ggplot(df1, aes(x = reorder(Job_Title, -avgApproval), y = avgApproval)) +
  geom_point() +
  theme_bw() +
  labs(title = "Job Type vs. Approval Rate") +
  geom_smooth(formula = y ~ x, method = "lm") +
  theme(axis.text.x = element_text(angle = 45, hjust = 0.5))




####################################################################################################
########################################    Modeling    ############################################
####################################################################################################


###########################  1. Data Preparation  ###########################
#############################################################################

### correlation among variables
#Change gender to dummy 
application$Applicant_Gender<-gsub(" ","",application$Applicant_Gender)
application$Applicant_Gender<- with(application, ifelse(application$Applicant_Gender=='M', 1, 0))

correlation<-subset(application, select=c("Applicant_Gender","Owned_Car", "Owned_Realty","Total_Children", "Total_Income",
                                          "Owned_Work_Phone","Owned_Phone","Owned_Email","Total_Family_Members","Applicant_Age","Years_of_Working",
                                          "Status"))

corr <- round(cor(correlation),2)  
head(corr)

library(reshape2)
# reduce the size of correlation matrix
melted_corr<- melt(corr)
# head(melted_corr_mat)

# plotting the correlation heatmap
library(ggplot2)
ggplot(data = melted_corr, aes(x=Var1, y=Var2,
                               fill=value)) + 
  geom_tile()+geom_text(aes(Var2, Var1, label = value), 
                        color = "black", size = 4)



# Inspect the data balance -> imbalance
table(application$Status)
prop.table(table(application$Status))

# balance our data: reduce half of approval data
ct_0 <- application_train %>% 
  filter(Status == 1) %>% 
  slice(-c(1:9000))

ct_1 <- application_train %>% 
  filter(Status == 0)

application_train <- rbind(ct_0, ct_1)

library(caret)
set.seed(129)
application_train$Status <- as.factor(application_train$Status)
ups_train <- upSample(x = application_train %>% select(-Status),
                      y = application_train$Status,
                      yname = "Status")

table(ups_train$Status)
#     0    1 
#  9751 9751       #(balance)




########################  2. Logistic Regression  ###########################
#############################################################################
ups_train$Status<-factor(ups_train$Status)
model_all <- glm(Status ~ .-Total_Children-Applicant_ID-Owned_Mobile_Phone -Total_Bad_Debt-Total_Good_Debt, ups_train, family = "binomial")
summary(model_all)

#Cross Validation 
set.seed(12345678)
train_control <- trainControl(method = "cv", number =5)

KFold_model_all<- train(Status ~ .-Applicant_ID-Owned_Mobile_Phone -Total_Bad_Debt-Total_Good_Debt, data =ups_train,
                        method = "glm",metric='Accuracy', trControl = train_control)
KFold_model_all  #Accuracy:0.6773158   Kappa:0.3546296




#######################  3. Step Logistic Regression  ######################
############################################################################

model_step <- step(glm(Status ~ .-Total_Children-Applicant_ID-Owned_Mobile_Phone -Total_Bad_Debt-Total_Good_Debt, ups_train, family = "binomial"), direction = "backward")
summary(model_step)

#> summary(model_step)     the variables step choose are as below
#Call:
#  glm(formula = Status ~ Applicant_Gender + Owned_Car + Owned_Realty + 
#       Total_Children + Total_Income + Income_Type + Education_Type + 
#        Family_Status + Housing_Type + Owned_Phone + Job_Title + 
#        Total_Family_Members + Applicant_Age + Years_of_Working, 
#        family = "binomial", data = ups_train)

set.seed(12345678)
train_control <- trainControl(method = "cv", number =5)


KFold_model_step <- train(Status ~ Applicant_Gender + Owned_Car + Owned_Realty + 
                            Total_Children + Total_Income + Income_Type + Education_Type + 
                            Family_Status + Housing_Type + Owned_Phone + Job_Title + 
                            Total_Family_Members + Applicant_Age + Years_of_Working, data = ups_train,
                          method = "glm",
                          metric = "Accuracy",
                          trControl = train_control)
KFold_model_step   # Accuracy: 0.6834176   Kappa: 0.3668343




###########################  4. Random Forest  ##############################
#############################################################################

library(randomForest)

rf<-randomForest(Status ~ .-Total_Children-Applicant_ID-Owned_Mobile_Phone -Total_Bad_Debt-Total_Good_Debt, 
                 data=ups_train, nodesize=5, ntree=500, mtry=4, proximity=TRUE)
rf

KFold_model_rf<- train(Status ~ .-Total_Children-Applicant_ID-Owned_Mobile_Phone-Total_Bad_Debt-Total_Good_Debt, data =ups_train,
                       method = "rf",trControl = train_control)
# KFold_model_rf will take more than 5 minutes to get the results
KFold_model_rf


varImp(rf)


# application_test    prediction
predictions <- predict(rf, newdata=application_test)
predictions


application_test$Status<-factor(application_test$Status)
confusionMatrix(data=predictions, application_test$Status)







# create bar chart to show Accuracy of the three models

results <- data.frame(Model = c("model_all", "model_step", "rf"),
                      Accurate = c(KFold_model_all$results$Accuracy[1], 
                             KFold_model_step$results$Accuracy[1], 
                             KFold_model_rf$results$Accuracy[1]))

ggplot(results, aes(x = Model, y = Accurate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(x = "Model", y = "Accurate", title = "                        5-fold Cross Validation Accuracy") +
  theme_minimal() +
  theme(panel.grid = element_blank())

