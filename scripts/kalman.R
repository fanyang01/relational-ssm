library(forecast)
library(tseries)
library(dlm)

args <- commandArgs(TRUE)
filename <- args[1]
# filename <- 'data.csv'

dataset <- read.csv(filename, header=FALSE)
dataset <- t(as.matrix(dataset))
series <- ts(dataset)


dim_observ = 32
dim_state = 8

F_size <- dim_observ * dim_state
G_size <- dim_state * dim_state

build_ssm <- function(theta) {	
	F <- matrix(theta[1:F_size], nrow = dim_observ, ncol = dim_state)
	G <- matrix(theta[(F_size+1):(F_size+G_size)], nrow = dim_state, ncol = dim_state)
	W <- exp(theta[F_size+G_size+1]) * diag(dim_state)
	V <- exp(theta[F_size+G_size+2]) * diag(dim_observ)
	x0 <- rep(0, dim_state)
	C0 <- exp(theta[F_size+G_size+3]) * diag(dim_state)
	return(dlm(m0 = x0, C0 = C0, FF = F, V = V, GG = G, W = W))
}

rand_init_theta <- c(runif(F_size + G_size, -1, 1), rep(0, 3))

learned_model <- dlmMLE(
	series[,1:dim_observ], rand_init_theta, build_ssm,
	method = 'L-BFGS-B', lower = -2.0, upper = 2.0, control = list(maxit = 10),
	debug = TRUE
)