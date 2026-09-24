// The model of logistic.stan, with the likelihood parallelized over slices of
// the observations via reduce_sum() (Stan's within-chain multithreading).
functions {
  real partial_loglik_lpmf(array[] int y_slice, int start, int end, matrix X, vector beta) {
    return bernoulli_logit_glm_lupmf(y_slice | X[start:end], 0, beta);
  }
}
data {
  int<lower=0> n;
  int<lower=0> p;
  matrix[n, p] X;
  array[n] int<lower=0, upper=1> y;
  int<lower=1> grainsize;
}
parameters {
  vector[p] beta;
}
model {
  beta ~ std_normal();
  target += reduce_sum(partial_loglik_lupmf, y, grainsize, X, beta);
}
