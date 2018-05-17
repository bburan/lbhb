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
    real<lower=0> a_scale;
    vector[n_conds] a_cond_delta;

    real<lower=0> b;

    real gt_loc;
    real gt_scale;
    vector[n_conds] gt_cond_delta;
    real<lower=0, upper=1> l;
}

transformed parameters {
    vector[n_conds] a_cond;
    //vector[n_conds] b_cond;

    real g_loc;
    vector[n_conds] g_cond;

    a_cond = a_loc + a_cond_delta * a_scale;
    //b_cond = b_loc + b_cond_delta * b_scale;

    g_loc = inv_logit(gt_loc);
    g_cond = inv_logit(gt_loc + gt_cond_delta * gt_scale);
}

model {
    vector[n_data] theta;
    vector[n_data] a;
    vector[n_data] g;

    l ~ beta(2, 20);

    a_loc ~ normal(-1, 2);
    a_scale ~ normal(0, 2)T[0,];
    a_cond_delta ~ normal(0, 1);

    b ~ gamma(1, 4);

    gt_loc ~ normal(-2.5, 2);
    gt_scale ~ normal(0, 1)T[0,];
    gt_cond_delta ~ normal(0, 1);

    a = a_cond[i_cond];
    g = g_cond[i_cond];

    theta = g + (1-g-l) ./ (1+exp(-(x-a)./b));
    k ~ binomial(n, theta);
}
