# Logistic Regression

In this vignette, we will implement logistic regression from scratch.

## Problem

Logistic regression models the probability that an observation belongs
to a particular class. Given a feature matrix \\X\\ and a weight vector
\\\beta\\, the predicted probability is:

\\P(y = 1 \| X) = \sigma(X \beta + \alpha)\\

where \\\sigma\\ is the logistic function \\\sigma(z) = \frac{1}{1 +
e^{-z}}\\ and \\\alpha\\ is the bias term.

In practice, logistic regression is typically fitted using Iteratively
Reweighted Least Squares (IRLS), which is a form of Newton’s method.
Here, we will use a simple gradient descept algorithm for didactic
purposes.

## Data

We’ll use the classic Titanic dataset from base R, which contains
survival data for the most famous shipwreck in history. The `Titanic`
object is a contingency table, so we first expand it into individual
observations.

``` r

library(anvl)
set.seed(42)

# Expand contingency table to individual observations
titanic_df <- as.data.frame(Titanic)
titanic <- titanic_df[rep(seq_len(nrow(titanic_df)), titanic_df$Freq), 1:4]
rownames(titanic) <- NULL

# Create design matrix with dummy variables
X <- model.matrix(~ Class + Sex + Age, data = titanic)[, -1]
y <- as.integer(titanic$Survived == "Yes")

# Standardize features for better gradient descent convergence
X <- scale(X)

n <- nrow(X)
p <- ncol(X)
```

The dataset contains 2201 observations and 5 predictor variables. Our
goal is to predict survival (`Survived`) based on three categorical
features:

- **Class**: Passenger class (1st, 2nd, 3rd, or Crew) - encoded as dummy
  variables with 1st class as the reference category
- **Sex**: Male or Female - encoded with Male as the reference category
- **Age**: Child or Adult - encoded with Child as the reference category

``` r

summary(titanic)
```

    ##   Class         Sex          Age       Survived  
    ##  1st :325   Male  :1731   Child: 109   No :1490  
    ##  2nd :285   Female: 470   Adult:2092   Yes: 711  
    ##  3rd :706                                        
    ##  Crew:885

The general survival rate was 32.30%.

Now we convert the data to `AnvlArray`s.

``` r

X_tensor <- nv_array(X, dtype = "f32")
y_tensor <- nv_array(y, dtype = "f32", shape = c(n, 1L))
```

## Model

The logistic regression model consists of computing the linear
combination of features and then applying the logistic function.

``` r

predict_proba <- function(X, beta, alpha) {
  logits <- X %*% beta + alpha
  nv_logistic(logits)
}
```

For binary classification, we use the binary cross-entropy loss:

\\\mathcal{L} = -\frac{1}{n} \sum\_{i=1}^{n} \left\[ y_i \log(p_i) +
(1 - y_i) \log(1 - p_i) \right\]\\

We need to be careful with numerical stability when computing the
logarithm of probabilities close to 0 or 1. We’ll add a small epsilon to
avoid taking the log of exactly 0.

``` r

binary_cross_entropy <- function(y_true, y_pred) {
  eps <- 1e-7
  y_pred_clipped <- nv_clamp(eps, y_pred, 1 - eps)
  loss <- -(y_true * log(y_pred_clipped) + (1 - y_true) * log(1 - y_pred_clipped))
  mean(loss)
}
```

We combine the prediction and loss computation into a single function.

``` r

model_loss <- function(X, y, beta, alpha) {
  y_pred <- predict_proba(X, beta, alpha)
  binary_cross_entropy(y, y_pred)
}
```

Using {anvl}’s automatic differentiation, we can obtain the gradients of
the loss with respect to the model parameters.

``` r

model_loss_grad <- gradient(model_loss, wrt = c("beta", "alpha"))
```

## Training

We will implement the training loop using
[`nv_while()`](https://r-xla.github.io/anvl/reference/nv_while.md). This
keeps the entire training loop within a single compiled function, which
is more efficient than repeatedly calling a JIT-compiled function from
R, especially for small models.

``` r

fit_logreg <- jit(function(X, y, beta, alpha, n_epochs, lr) {
  output <- nv_while(
    list(beta = beta, alpha = alpha, epoch = nv_scalar(0L)),
    \(beta, alpha, epoch) epoch < n_epochs,
    \(beta, alpha, epoch) {
      grads <- model_loss_grad(X, y, beta, alpha)
      list(
        beta = beta - lr * grads$beta,
        alpha = alpha - lr * grads$alpha,
        epoch = epoch + 1L
      )
    }
  )
  list(beta = output$beta, alpha = output$alpha)
})
```

We initialize the parameters and train the model with a single function
call.

``` r

beta_init <- nv_array(rnorm(p), dtype = "f32", shape = c(p, 1L))
alpha_init <- nv_scalar(0, dtype = "f32")

result <- fit_logreg(
  X_tensor, y_tensor,
  beta_init, alpha_init,
  nv_scalar(50000L),
  nv_scalar(0.1)
)

result
```

    ## $beta
    ## AnvlArray
    ##  -0.3419
    ##  -0.8300
    ##  -0.4206
    ##   0.9920
    ##  -0.2304
    ## [ CPUf32{5,1} ] 
    ## 
    ## $alpha
    ## AnvlArray
    ##  -0.8538
    ## [ CPUf32{} ]

Let’s verify our implementation by comparing with R’s built-in
[`glm()`](https://rdrr.io/r/stats/glm.html).

``` r

glm_fit <- glm(y ~ X, family = binomial)
```

Now let’s compare the coefficients:

    ##     Parameter       anvl        glm
    ## 1 (Intercept) -0.8538059 -0.8538077
    ## 2    Class2nd -0.3418888 -0.3418906
    ## 3    Class3rd -0.8299901 -0.8299946
    ## 4   ClassCrew -0.4206346 -0.4206313
    ## 5   SexFemale  0.9919737  0.9919786
    ## 6    AgeAdult -0.2303528 -0.2303616

The estimates are very close, confirming that our gradient descent
implementation converges to the same solution as
[`glm()`](https://rdrr.io/r/stats/glm.html).
