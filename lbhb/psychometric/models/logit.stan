data {
    int<lower=1> n_data;
    int<lower=1> n[n_data];
    int<lower=0> k[n_data];
    vector[n_data] x;
}

parameters {
    real a;
    real<lower=0> b;
    real g;
    real l;

}

model {
    vector[n_data] theta;

    g ~ beta(2, 20);
    l ~ beta(2, 20);
    a ~ normal(-1, 2);
    b ~ gamma(1, 4);

    theta = g + (1-g-l) ./ (1+exp(-(x-a)./b));
    k ~ binomial(n, theta);
}
