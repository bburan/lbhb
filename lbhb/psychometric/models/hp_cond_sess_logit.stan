data {
    int<lower=1> n_data;
    int<lower=1> n[n_data];
    int<lower=0> k[n_data];
    vector[n_data] x;
	vector[n_data] cond;		  // actual condition value
    int<lower=1> n_sess;          // number of sessions
    int<lower=0> i_sess[n_data];  // index of session
}

parameters {
	real a_loc;
	real a_cond_slope;
    real<lower=0> a_sess_scale;
    vector[n_sess] a_sess_delta;

    real gt_loc;
	real gt_cond_slope;
    real<lower=0> gt_sess_scale;
    vector[n_sess] gt_sess_delta;

    real bt_loc;
    real bt_cond_slope;
    real<lower=0> bt_sess_scale;
	vector[n_sess] bt_sess_delta;

    real<lower=0, upper=1> l;
}

transformed parameters {
}

model {
    vector[n_data] theta;
    vector[n_data] a;
    vector[n_data] b;
    vector[n_data] g;

    l ~ beta(2, 20);

    a_loc ~ normal(-1, 2);
	a_cond_slope ~ normal(0, 1);

    a_sess_scale ~ normal(0, 0.1)T[0,];
    a_sess_delta ~ normal(0, 1);

    bt_loc ~ normal(-1.5, 1);
	bt_cond_slope ~ normal(0, 1);
    bt_sess_scale ~ normal(0, 0.1)T[0,];
	bt_sess_delta ~ normal(0, 1);

    gt_loc ~ normal(-2.5, 2);
	gt_cond_slope ~ normal(0, 1);
    gt_sess_scale ~ normal(0, 0.1)T[0,];
    gt_sess_delta ~ normal(0, 1);

	a = a_loc + a_cond_slope * cond + a_sess_delta[i_sess] * a_sess_scale;
    b = exp(bt_loc + bt_cond_slope * cond + bt_sess_delta[i_sess] * bt_sess_scale);
    g = inv_logit(gt_loc + gt_cond_slope * cond + gt_sess_delta[i_sess] * gt_sess_scale);

    theta = g + (1-g-l) ./ (1+exp(-(x-a)./b));
    k ~ binomial(n, theta);
}
