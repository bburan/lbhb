data {
    int<lower=1> n_data;
    int<lower=1> n[n_data];
    int<lower=0> k[n_data];
    vector[n_data] x;

    int<lower=1> n_conds;          // number of conds
    int<lower=0> i_cond[n_data];   // index of cond
}

parameters {
    real a_loc;
    real a_scale;
    vector[n_conds] a_cond_delta;

    real<lower=0> b;
    real g;
    real l;
}

transformed parameters {
    vector[n_conds] a_cond;

    a_cond = a_loc + a_cond_delta * a_scale;
}

model {
    vector[n_data] theta;
    vector[n_data] a;

    g ~ beta(2, 20);
    l ~ beta(2, 20);
    b ~ gamma(1, 4);

    a_loc ~ normal(-1, 2);
    a_scale ~ normal(0, 2)T[0,];
    a_cond_delta ~ normal(0, 1);

    a = a_cond[i_cond];

    theta = g + (1-g-l) ./ (1+exp(-(x-a)./b));
    k ~ binomial(n, theta);
}
