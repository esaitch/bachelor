// Biased Nash model for the simulated data

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

model{
  // the probability model and priors 
  vector[N] theta; // a vector for the estimated parameter (probability of success)
  thetaG ~ normal(0.5,0.1); // prior for the group parameter (mean, sd)
  thetaP ~ normal(thetaG,0.2); // prior for the subject parameter. (mean, sd). Mean is taken from group prior 
  
  for(n in 1 : N) { // loop through observations
    theta[n] = thetaP[Subject[n]]; //Fill vector theta with estimated probability of success parameter
  }
  y ~ bernoulli_logit(theta); // the probability model. bernoulli_logit is a special case of the binomial distribution 
}
