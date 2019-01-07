setwd('/Users/Zachary/playground/CS498/HW6')
getwd()

library("base")
library("trafo")

# read the data and extract out measurements and housing price
housingData<-read.table('housing.data.txt')

# extract dependant variable
housePrice<-housingData[,14]

# regress house price against all others
relation<-lm(housePrice~.-V14,data=housingData)

# plot residuals vs leverage vs cook's distance
png(filename='output/orig_diagnostic_plot.png')
plot(relation, which=c(5), id.n=10)
dev.off()

# plot residuals vs fitted values
png(filename='output/res_fitted.png')
plot(relation$fitted.values, rstandard(relation),
     xlab="Fitted values", ylab="Standard residuals",
     main="Standard residuals against fitted values\n original depedent variable")
dev.off()

# remove possible outliers
# to_removed<-c(-365, -369, -372, -373)
# to_removed<-c(-365, -369,  -372, -373, -366, -370)
# to_removed<-c(-365, -369,  -372, -373, -366, -370, -368, -371, -413)
# to_removed<-c(-365, -369,  -372, -373, -366, -370, -368, -371, -381)
to_removed<-c(-365, -369,  -372, -373, -366, -370, -368, -371)
rm_housingData<-housingData[to_removed,]

# extract dependant variable
rm_housePrice<-rm_housingData[,14]

# regress house price against all others after removing outliers
rm_relation<-lm(rm_housePrice~.-V14, data=rm_housingData)

# plot residuals vs leverage vs cook's distance
png(filename='output/rm_diagnostic_plot.png')
plot(rm_relation, which=c(5), id.n=10)
dev.off()

# Box-Cox transformation
png(filename='output/orig_box_plot.png')
boxcox_list<-boxcox(rm_relation)
dev.off()

# get the best value of parameter
best_lamdba<-boxcox_list$lambdahat

# transform depedent variable
transf_rm_housePrice<-boxcoxTransform(rm_housePrice, best_lamdba)

# regress house price against all others after removing outliers
# and transforming depedeng variable
transf_rm_relation<-lm(transf_rm_housePrice~.-V14, data=rm_housingData)

# plot standard residuals vs leverage vs cook's distance
png(filename='output/transf_rm_diagnostic_plot.png')
plot(transf_rm_relation, which=c(5), id.n=10)
dev.off()
# plot standard residuals vs fitted values
png(filename='output/transf_rm_res_fitted.png')
plot(transf_rm_relation$fitted.values, rstandard(transf_rm_relation),
     xlab="Fitted values", ylab="Standard residuals",
     main="Standard residuals against fitted values\n transformed depedent variable")
dev.off()

# transfrom our predicted values back to get the final prediction
final_prediction<-lapply(transf_rm_relation$fitted.values, function(x) (x*best_lamdba + 1)^(1/best_lamdba))

# final plot of Fitted house price vs True house price
png(filename='output/final_plot.png')
plot(rm_housePrice, final_prediction,
     xlab="True house price", ylab="Predicted house price",
     main="Predicted vs True house prices for\n a regression of house price against all variables")
dev.off()
