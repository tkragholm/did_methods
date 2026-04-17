use faer::Mat;

#[must_use]
pub fn ipt_loss_grad_hess(
    design: &Mat<f64>,
    target: &Mat<f64>,
    beta: &Mat<f64>,
    vstar: f64,
) -> (f64, Mat<f64>, Mat<f64>) {
    let n_obs = design.nrows();
    let n_cols = design.ncols();
    let x_beta = design * beta;
    let exp_vstar = (-vstar).exp();
    let an = -vstar - exp_vstar;
    let bn = -1.0 + exp_vstar;
    let cn = -exp_vstar;

    let mut loss = 0.0;
    let mut gradient = Mat::zeros(n_cols, 1);
    let mut hessian = Mat::zeros(n_cols, n_cols);

    for i in 0..n_obs {
        let v_lin = *x_beta.get(i, 0);
        let d_i = *target.get(i, 0);
        let x_row = design.row(i);

        let (phi, phi_p, phi_pp) = if v_lin < vstar {
            let ev = (-v_lin).exp();
            (-v_lin - ev, -1.0 + ev, -ev)
        } else {
            (
                (0.5 * cn * v_lin).mul_add(v_lin, bn * v_lin) + an,
                cn.mul_add(v_lin, bn),
                cn,
            )
        };

        loss -= (1.0 - d_i).mul_add(phi, d_i * v_lin);
        let grad_scalar = (-(1.0 - d_i)).mul_add(phi_p, d_i);
        let hess_scalar = -(1.0 - d_i) * phi_pp;
        for j in 0..n_cols {
            *gradient.get_mut(j, 0) += grad_scalar * x_row.get(j);
        }
        for j in 0..n_cols {
            for k in 0..n_cols {
                *hessian.get_mut(j, k) += hess_scalar * x_row.get(j) * x_row.get(k);
            }
        }
    }
    (loss, gradient, hessian)
}
