data {
  int<lower=1> N;
  int<lower=1> T;
  real S_0;
  real I_0;
  real<lower=0> R_0_min;
  real<lower=0> R_0_max;
  real<lower=0> D_min;
  real<lower=0> D_max;
  real<lower=0> alpha_min;
  real<lower=0> alpha_max;
  array[T] int<lower=0> cases;
}
parameters {
  real<lower=R_0_min,upper=R_0_max> R_0;
  real<lower=D_min,upper=D_max> D;
  real<lower=alpha_min,upper=alpha_max> alpha;
}
transformed parameters {
  real beta = R_0 * (1 / D);
  real gamma = 1 / D;
}
model {
  // Likelihood
  array[T] real S;
  array[T] real I;
  array[T] real R;

  S[1] = S_0;
  I[1] = I_0;
  R[1] = 0;
  cases[1] ~ poisson(N * alpha * I[1]);

  for (t in 2 : T) {
    real new_infections = beta * S[t - 1] * I[t - 1];
    real new_recoveries = gamma * I[t - 1];

    S[t] = S[t - 1] - new_infections;
    I[t] = I[t - 1] + new_infections - new_recoveries;
    R[t] = R[t - 1] + new_recoveries;

    cases[t] ~ poisson(N * alpha * new_infections);
  }

  // Prior
  R_0 ~ uniform(R_0_min, R_0_max);
  D ~ uniform(D_min, D_max);
  alpha ~ uniform(alpha_min, alpha_max);
}
