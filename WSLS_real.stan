// WSLS model for experimental data (including posterior predictions)

data {
  //define the data that must be passed to Stan from the environment 
  int N; //number of observations/trials - N*P
  int P; //number of participants
  int right[N]; //choice of hand
  int subject[N]; //participant id
  int success[N]; //correct or incorrect choice
}

parameters {
  // Define the parameters that we will estimate + restrictions 
  vector[P] thetaP; //probability, one for each participant
  vector<lower=0,upper=1>[P] stayBiasP; //the probability depends on biases. One for each participant
  vector<lower=0,upper=1>[P] shiftBiasP;
  real stayBiasG; // stayBias gets mean from group
  real shiftBiasG; // shiftBias gets mean from group
  real<lower=0> biasVariation; // the sd
}

transformed parameters {
  // saves the theta so we can use it both in model and generated quantities (not necessary for simulated data as it takes more space and time)
  vector[N] theta; //a theta for each observation - set N=N*P
  
  for(n in 1 : N) { //loop through all observations
    if (n==1) { //if first trial
      theta[n] = 0.5; //random probability of success
    }
    else if (right[n-1]==1 && success[n-1]==1) { //if previous choice was right hand and correct
      theta[n] = stayBiasP[subject[n]]; //probability of success is the stay bias
    }
    else if (right[n-1] == 1 && success[n-1] == 0) { //if previous choice was right hand and incorrect
      theta[n] = 1 - shiftBiasP[subject[n]]; //probability of success is 1-shift bias
    }
    else if (right[n-1] == 0 && success[n-1] == 0) { //if previous choice was left hand and incorrect
      theta[n] = shiftBiasP[subject[n]]; // probability of success is the shift bias
    }
    else if (right[n-1] == 0 && success[n-1] == 1) { //if previous choice was left hand and incorrect
      theta[n] =  1-stayBiasP[subject[n]]; //probability of success is 1-stay bias
    }
  }
}

model{
  // the probability model and priors 
  biasVariation ~ normal(0,0.3); // variation prior (mean, sd)
  stayBiasG ~ normal(0.73,0.1); // stay bias for group (mean, sd)
  shiftBiasG ~ normal(0.73,0.1); // shift bias for group (mean, sd)
  stayBiasP ~ normal(stayBiasG,biasVariation); // individual stay bias (mean, sd)
  shiftBiasP ~ normal(shiftBiasG,biasVariation); // individual shift bias (mean, sd)
 
  right ~ bernoulli_logit(theta); // the probability model. bernoulli_logit is a special case of the binomial distribution 
}

generated quantities{
  vector[N] log_lik; //vector for log likelihood. Used for LOO comparison
  vector[N] y_sim; //vector for draws from the posterior predictions
  for(n in 1:N) log_lik[n] = bernoulli_logit_lpmf(right[n] | theta[n]); //save the log likelihood. lpmf is the mass function
  for(n in 1:N) y_sim[n] = bernoulli_logit_rng(theta[n]); //generate posterior predictive draws (yhat). rng is the random number generator function (generating numbers from the posterior predictive)
}

