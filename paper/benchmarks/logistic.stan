data {
  int<lower=0> n;
  int<lower=0> p;
  matrix[n, p] X;
  array[n] int<lower=0, upper=1> y;
}
parameters {
  vector[p] beta;
}
model {
  beta ~ std_normal();
  y ~ bernoulli_logit_glm(X, 0, beta);
}
