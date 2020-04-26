library(forecast)
library(tseries)


args <- commandArgs(TRUE)

dataset <- read.csv(args[1], header=FALSE)
dataset <- t(as.matrix(dataset))
series <- ts(dataset)
num_series <- dim(series)[1]
for (i in 1:num_series) {
	fit <- auto.arima(series[,i], seasonal=FALSE)
	print(fit$loglik)
}

