data {
    int<lower=1> n_data;
    int<lower=1> n[n_data];
    int<lower=0> k[n_data];
    vector[n_data] x;
	vector[n_data] cond;		  // actual condition value
}

parameters {
	real a_loc;
	real a_cond_slope;

    real gt_loc;
	real gt_cond_slope; 
    real bt_loc;

    real<lower=0, upper=1> l;
}

model {
    vector[n_data] theta;
    vector[n_data] a;
    //vector[n_data] b;
    vector[n_data] g;
    real b;

    l ~ beta(2, 20);

    a_loc ~ normal(-1, 2);
	a_cond_slope ~ normal(0, 1);

    bt_loc ~ normal(-1.5, 1);

    gt_loc ~ normal(-1.5, 1);
	gt_cond_slope ~ normal(0, 1);

	a = a_loc + a_cond_slope * cond;
    b = exp(bt_loc);
    g = inv_logit(gt_loc + gt_cond_slope * cond);

    theta = g + (1-g-l) ./ (1+exp(-(x-a)./b));
    k ~ binomial(n, theta);
}
