data {
    int<lower=1> n_data;
    int<lower=1> n[n_data];
    int<lower=0> k[n_data];
    vector[n_data] x;

    int<lower=1> n_cond;          // number of conditions
    int<lower=0> i_cond[n_data];  // index of condition

    int<lower=1> n_sess;          // number of sessions
    int<lower=0> i_sess[n_data];  // index of session
}

parameters {
    real a_loc;
    real<lower=0> a_cond_scale;
    real<lower=0> a_sess_scale;
    vector[n_cond] a_cond_delta;
    vector[n_sess] a_sess_delta;

    real gt_loc;
    real<lower=0> gt_cond_scale;
    //real<lower=0> gt_sess_scale;
    vector[n_cond] gt_cond_delta;
    //vector[n_sess] gt_sess_delta;

    real<lower=0> b;
    real<lower=0, upper=1> l;
}

transformed parameters {
    vector[n_cond] a_cond;
    vector[n_sess] a_sess;
    real g_loc;
    vector[n_cond] g_cond;

    a_cond = a_loc + a_cond_delta * a_cond_scale;
    a_sess = a_loc + a_sess_delta * a_sess_scale;

    g_cond = inv_logit(gt_loc + gt_cond_delta * gt_cond_scale);
    g_loc = inv_logit(gt_loc);
}

model {
    vector[n_data] theta;
    vector[n_data] a;
    vector[n_data] g;

    l ~ beta(2, 20);

    a_loc ~ normal(-1, 2);
    a_cond_scale ~ normal(0, 0.5)T[0,];
    a_sess_scale ~ normal(0, 0.5)T[0,];
    a_cond_delta ~ normal(0, 1);
    a_sess_delta ~ normal(0, 1);

    b ~ gamma(1, 4);

    gt_loc ~ normal(-2.5, 2);
    gt_cond_scale ~ normal(0, 1)T[0,];
    gt_cond_delta ~ normal(0, 1);
    //gt_sess_scale ~ normal(0, 1)T[0,];
    //gt_sess_delta ~ normal(0, 1);

    a = a_loc + a_cond_delta[i_cond] * a_cond_scale + a_sess_delta[i_sess] * a_sess_scale;
    g = g_cond[i_cond];

    theta = g + (1-g-l) ./ (1+exp(-(x-a)./b));
    k ~ binomial(n, theta);
}
