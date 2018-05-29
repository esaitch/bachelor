// Biased Nash model for experimental data (including posterior predictions)

data {
  //define the data that must be passed to Stan from the environment 
  int N; //number of observations/trials - N*P
  int P; //number of participants
  int y[N]; //outcome
  int Subject[N];
}

parameters {
  // Define the parameters that we will estimate + restrictions 
  vector[P] thetaP; //probability of success parameter
  real thetaG; //<lower=0,upper=1>; //the group parameter. (Bound between 0 and 1)
}

transformed parameters {
  // saves the theta so we can use it both in the model and generated quantities (not necessary for simulated data as it takes more space and time)
  vector[N] theta; // Create vector
  
  for(n in 1 : N) { // loop through observations
    theta[n] = thetaP[Subject[n]]; //save the participant theta in the vector
  }
}

model{
  // the probability model and priors 
  thetaG ~ normal(0.5,0.1); // prior for the group parameter (mean, sd).
  thetaP ~ normal(thetaG,0.2); // prior for the subject parameter. (mean, sd). Mean is taken from group
  y ~ bernoulli_logit(theta); // the probability model. bernoulli_logit is a special case of the binomial distribution 
}

generated quantities{
  vector[N] log_lik; //vector for log likelihood. Used for LOO comparison
  vector[N] y_sim; //vector for draws from the posterior predictions
  
  for(n in 1:N) log_lik[n] = bernoulli_logit_lpmf(y[n] | theta[n]); //save the log likelihood. lpmf is the mass function
  
  for(n in 1:N) y_sim[n] = bernoulli_logit_rng(theta[n]); //generate posterior predictive draws (yhat). rng is the random number generator function (generating numbers from the posterior predictive)
}

