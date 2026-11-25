// The logistic regression model of logistic.stan as a TMB template.
// parallel_accumulator lets TMB evaluate the likelihood with several OpenMP
// threads (see TMB::openmp()); with one thread it is an ordinary sum.
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() () {
  DATA_MATRIX(X);
  DATA_VECTOR(y);
  PARAMETER_VECTOR(beta);
  parallel_accumulator<Type> nll(this);
  for (int i = 0; i < X.rows(); i++) {
    Type eta = (X.row(i).transpose().array() * beta.array()).sum();
    nll -= y(i) * eta - log(Type(1) + exp(eta));
  }
  nll += Type(0.5) * (beta * beta).sum();
  return nll;
}
