import os
import concurrent.futures
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm  # 用于显示进度条

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


# ============================================================
# 1. 数据读取与预处理
# ============================================================

def load_and_preprocess(path):
    """
    读取 CSV，按 sample_id 和 row 对 recall 求平均，
    并转换为退化量 X = 1 - recall。
    """
    df = pd.read_csv(path)
    required_cols = {"sample_id", "row", "recall"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    df_mean = (
        df.groupby(["sample_id", "row"], as_index=False)["recall"]
        .mean()
        .sort_values(["sample_id", "row"])
        .reset_index(drop=True)
    )
    df_mean["X"] = 1.0 - df_mean["recall"]
    return df_mean


def build_data_matrices(df_mean):
    """
    从预处理后的 df_mean 构建时间网格和观测矩阵
    """
    t = np.sort(df_mean["row"].unique())
    samples = df_mean["sample_id"].unique()

    X_list = []
    for sid in samples:
        arr = (
            df_mean[df_mean["sample_id"] == sid]
            .sort_values("row")["X"]
            .values
        )
        if len(arr) != len(t):
            raise ValueError(
                f"样本 {sid} 的时间点数与公共时间网格不一致，"
                "训练集各样本应共享相同的 row 网格。"
            )
        X_list.append(arr)

    return t, X_list, samples


# ============================================================
# 2. MLE 拟合 Theta1 = {theta, gamma, mu0, sigma0, sigma}
# ============================================================

def Lambda_vec(t, theta):
    return np.log1p(theta * t)

def tau_vec(t, gamma):
    return np.power(t, gamma)

def neg_log_likelihood(params, t, X_list):
    log_theta, log_gamma, mu0, log_sigma0, log_sigma = params

    theta = np.exp(log_theta)
    gamma = np.exp(log_gamma)
    sigma0 = np.exp(log_sigma0)
    sigma = np.exp(log_sigma)

    lam = Lambda_vec(t, theta)
    tau = tau_vec(t, gamma)
    tau_mat = np.minimum.outer(tau, tau)

    total_nll = 0.0
    n_t = len(t)
    jitter = 1e-8

    for X in X_list:
        Sigma = (
            sigma**2 * tau_mat
            + sigma0**2 * np.outer(lam, lam)
            + jitter * np.eye(n_t)
        )

        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            return np.inf

        diff = X - mu0 * lam
        sol = np.linalg.solve(Sigma, diff)

        total_nll += 0.5 * (
            n_t * np.log(2 * np.pi) + logdet + diff.dot(sol)
        )

    return total_nll

def fit_theta1_from_df(df_mean):
    t, X_list, _ = build_data_matrices(df_mean)

    init = np.array([
        np.log(0.01),  # log_theta
        np.log(1.0),   # log_gamma
        0.1,           # mu0
        np.log(0.1),   # log_sigma0
        np.log(0.1),   # log_sigma
    ])

    res = minimize(
        neg_log_likelihood,
        init,
        args=(t, X_list),
        method="L-BFGS-B"
    )

    if not res.success:
        raise RuntimeError(f"Theta1 拟合失败: {res.message}")

    log_theta, log_gamma, mu0, log_sigma0, log_sigma = res.x

    return {
        "theta": np.exp(log_theta),
        "gamma": np.exp(log_gamma),
        "mu0": mu0,
        "sigma0": np.exp(log_sigma0),
        "sigma": np.exp(log_sigma),
    }


# ============================================================
# 3. 贝叶斯估计 Q_lambda (目前在主流程中硬编码为 1e-6)
# ============================================================

def get_deltas(t, theta, gamma):
    Lam = np.log1p(theta * t)
    Tau = np.power(t, gamma)

    delta_L = np.zeros_like(t, dtype=float)
    delta_tau = np.zeros_like(t, dtype=float)

    delta_L[1:] = Lam[1:] - Lam[:-1]
    delta_tau[1:] = Tau[1:] - Tau[:-1]

    return delta_L, delta_tau

def bayesian_estimate_Q_numpyro(
    df_mean, theta, gamma, mu0, sigma0, sigma_B,
    num_warmup=500, num_samples=1000, rng_seed=0,
):
    samples = df_mean["sample_id"].unique()
    sample_data = {}

    # 1. 在数据准备阶段加上 tqdm 进度条
    print("\n>>> 正在准备贝叶斯 MCMC 所需数据...")
    for sid in tqdm(samples, desc="数据准备进度", unit="样本"):
        df_s = df_mean[df_mean["sample_id"] == sid].sort_values("row").reset_index(drop=True)
        t_vec = df_s["row"].values.astype(float)
        X_obs = df_s["X"].values.astype(float)
        delta_L, delta_tau = get_deltas(t_vec, theta, gamma)

        sample_data[str(sid)] = {
            "X_obs": X_obs, "delta_L": delta_L, "delta_tau": delta_tau,
        }

    def model(sample_data, mu0, sigma0, sigma_B):
        Q_lambda = numpyro.sample("Q_lambda", dist.InverseGamma(2.0, 1e-6))

        for sid, info in sample_data.items():
            X_obs = jnp.asarray(info["X_obs"])
            delta_L = jnp.asarray(info["delta_L"])
            delta_tau = jnp.asarray(info["delta_tau"])

            a0_s = numpyro.sample(f"a0_{sid}", dist.Normal(mu0, sigma0))
            eps_rw = numpyro.sample(
                f"eps_{sid}",
                dist.GaussianRandomWalk(scale=jnp.sqrt(Q_lambda), num_steps=len(X_obs))
            )
            a_vals = a0_s + eps_rw

            mu_X = X_obs[:-1] + a_vals[:-1] * delta_L[1:]
            sigma_X = sigma_B * jnp.sqrt(jnp.maximum(delta_tau[1:], 1e-12))

            numpyro.sample(
                f"Xobs_{sid}",
                dist.Normal(mu_X, sigma_X),
                obs=X_obs[1:]
            )

    print("\n>>> 启动 NumPyro MCMC 采样 (这可能需要一些时间)...")
    # 2. 将 progress_bar=False 改为 progress_bar=True
    # 注意：如果你想尝试多链并行，可以把 num_chains 改为 4（前提是你电脑核心够用）
    mcmc = MCMC(
        NUTS(model), 
        num_warmup=num_warmup, 
        num_samples=num_samples, 
        num_chains=1, 
        progress_bar=True  # <--- 在这里开启进度条
    )
    
    mcmc.run(
        jax.random.PRNGKey(rng_seed), 
        sample_data=sample_data, 
        mu0=mu0, 
        sigma0=sigma0, 
        sigma_B=sigma_B
    )
    
    posterior = mcmc.get_samples()
    return float(np.asarray(posterior["Q_lambda"]).mean())
# ============================================================
# 4. 卡尔曼滤波估计 lambda 序列
# ============================================================

def kalman_estimate_lambda(df_sample, theta, sigma_B, gamma, mu0, sigma0, R_meas, Q_lambda):
    df_s = df_sample.sort_values("row").reset_index(drop=True).copy()
    rows = df_s["row"].values.astype(float)
    X_obs = df_s["X"].values.astype(float)

    n = len(df_s)
    lambda_estimates = np.zeros(n)
    lambda_vars = np.zeros(n)

    z = np.array([X_obs[0], mu0], dtype=float)
    P = np.diag([1e-6, sigma0**2]).astype(float)
    H = np.array([[1.0, 0.0]], dtype=float)

    lambda_estimates[0] = z[1]
    lambda_vars[0] = P[1, 1]

    prev_t = rows[0]
    prev_L = np.log1p(theta * prev_t)
    prev_tau = prev_t**gamma

    for i in range(1, n):
        t = rows[i]

        cur_L = np.log1p(theta * t)
        delta_L = cur_L - prev_L

        cur_tau = t**gamma
        delta_tau = max(cur_tau - prev_tau, 0.0)

        F = np.array([
            [1.0, delta_L],
            [0.0, 1.0]
        ], dtype=float)

        Q_x = sigma_B**2 * delta_tau
        Q = np.diag([Q_x, Q_lambda]).astype(float)

        z_pred = F @ z
        P_pred = F @ P @ F.T + Q

        S = (H @ P_pred @ H.T)[0, 0] + R_meas
        K = (P_pred @ H.T) / S
        y_pred = (H @ z_pred)[0]

        innovation = X_obs[i] - y_pred
        z = z_pred + K.flatten() * innovation
        P = (np.eye(2) - K @ H) @ P_pred

        lambda_estimates[i] = z[1]
        lambda_vars[i] = max(P[1, 1], 1e-12)

        prev_t = t
        prev_L = cur_L
        prev_tau = cur_tau

    df_out = df_s.copy()
    df_out["lambda_est"] = lambda_estimates
    df_out["lambda_var"] = lambda_vars

    return df_out

def run_filter_on_test(df_test_mean, theta1, Q_lambda_est, R_meas):
    results = []
    for sid, df_s in df_test_mean.groupby("sample_id"):
        filtered = kalman_estimate_lambda(
            df_sample=df_s, theta=theta1["theta"], sigma_B=theta1["sigma"],
            gamma=theta1["gamma"], mu0=theta1["mu0"], sigma0=theta1["sigma0"],
            R_meas=R_meas, Q_lambda=Q_lambda_est,
        )
        results.append(filtered)
    return pd.concat(results, axis=0).reset_index(drop=True)


# ============================================================
# 5. 多核并行输出每个时间点的 RUL 
# ============================================================

def estimate_rul_pdf(
    t_now, X_now, lambda_mean, lambda_var, theta, gamma, sigma_B,
    failure_threshold, max_s=200.0, n_grid=1200, n_mc=5000, seed=42, # 注意这里改为了5000加速计算
):
    D_now = failure_threshold - X_now

    if D_now <= 0:
        return {
            "grid": np.array([0.0]), "pdf": np.array([1.0]),
            "rul_mean": 0.0, "rul_median": 0.0, "rul_mode": 0.0, "already_failed": 1,
        }

    s = np.linspace(1e-6, max_s, n_grid)
    Lambda = lambda t: np.log1p(theta * t)
    Tau = lambda t: t**gamma

    delta_L = Lambda(t_now + s) - Lambda(t_now)
    delta_T = np.maximum(Tau(t_now + s) - Tau(t_now), 1e-12)

    rng = np.random.default_rng(seed)
    a_samples = rng.normal(loc=lambda_mean, scale=np.sqrt(max(lambda_var, 1e-12)), size=n_mc)

    sigma2 = sigma_B**2

    pref = D_now / np.sqrt(2 * np.pi * sigma2 * delta_T**3)
    exp_term = np.exp(-((D_now - a_samples[:, None] * delta_L) ** 2) / (2 * sigma2 * delta_T))
    f_pdf = np.clip((pref * exp_term).mean(axis=0), 0.0, None)

    area = np.trapezoid(f_pdf, s)
    if area <= 0 or not np.isfinite(area):
        raise RuntimeError("RUL PDF 归一化失败，请检查参数或阈值设置。")

    f_pdf /= area
    cdf = np.cumsum(f_pdf) / np.sum(f_pdf)

    rul_mean = float(np.trapezoid(s * f_pdf, s))
    rul_mode = float(s[np.argmax(f_pdf)])
    rul_median = float(np.interp(0.5, cdf, s))

    return {
        "grid": s, "pdf": f_pdf,
        "rul_mean": rul_mean, "rul_median": rul_median, "rul_mode": rul_mode, "already_failed": 0,
    }


def _process_single_sample(args):
    """
    多进程 Worker：计算单个样本所有时间点的 RUL
    """
    sid, df_s, theta1, failure_threshold = args
    df_s = df_s.sort_values("row").reset_index(drop=True)
    
    results = []
    for _, row_data in df_s.iterrows():
        rul_res = estimate_rul_pdf(
            t_now=float(row_data["row"]),
            X_now=float(row_data["X"]),
            lambda_mean=float(row_data["lambda_est"]),
            lambda_var=float(row_data["lambda_var"]),
            theta=float(theta1["theta"]),
            gamma=float(theta1["gamma"]),
            sigma_B=float(theta1["sigma"]),
            failure_threshold=float(failure_threshold),
        )

        results.append({
            "sample_id": sid,
            "row": float(row_data["row"]),
            "recall": float(row_data["recall"]),
            "X": float(row_data["X"]),
            "lambda_est": float(row_data["lambda_est"]),
            "lambda_var": float(row_data["lambda_var"]),
            "rul_mean": rul_res["rul_mean"],
            "rul_median": rul_res["rul_median"],
            "rul_mode": rul_res["rul_mode"],
            "already_failed": int(rul_res["already_failed"]),
        })
        
    return results


def predict_rul_for_test_parallel(df_test_filtered, theta1, failure_threshold, max_workers=None):
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)
        
    print(f"\n🚀 启动多核并行计算 (分配核心数: {max_workers}) ...")

    tasks = [
        (sid, df_s, theta1, failure_threshold) 
        for sid, df_s in df_test_filtered.groupby("sample_id")
    ]

    all_results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_sample, task): task[0] for task in tasks}
        
        # 使用 tqdm 渲染进度条
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="计算进度", unit="样本"):
            try:
                sample_results = future.result()
                all_results.extend(sample_results)
            except Exception as e:
                sid = futures[future]
                print(f"\n❌ 样本 {sid} 计算发生异常: {e}")

    df_rul = pd.DataFrame(all_results).sort_values(["sample_id", "row"]).reset_index(drop=True)
    return df_rul


# ============================================================
# 6. 主流程
# ============================================================

def train_and_predict_rul(
    train_path, test_path,
    output_rul_path="rul_predictions.csv",
    output_lambda_path="test_lambda_paths.csv",
    failure_threshold=0.30,
    R_meas=0.005**2,
):
    print(">>> 开始训练集拟合 Theta1 ...")
    df_train_mean = load_and_preprocess(train_path)
    theta1 = fit_theta1_from_df(df_train_mean)

    
    Q_lambda_est = bayesian_estimate_Q_numpyro(
        df_mean=df_train_mean,
        theta=theta1["theta"],
        gamma=theta1["gamma"],
        mu0=theta1["mu0"],
        sigma0=theta1["sigma0"],
        sigma_B=theta1["sigma"],
        num_warmup=500,
        num_samples=1000,
        rng_seed=0,
    )

    
    print(">>> 开始测试集卡尔曼滤波 ...")
    df_test_mean = load_and_preprocess(test_path)
    df_test_filtered = run_filter_on_test(df_test_mean, theta1, Q_lambda_est, R_meas)

    df_rul = predict_rul_for_test_parallel(
        df_test_filtered=df_test_filtered,
        theta1=theta1,
        failure_threshold=failure_threshold,
        max_workers=None
    )

    df_test_filtered.to_csv(output_lambda_path, index=False)
    df_rul.to_csv(output_rul_path, index=False)

    print("\n✅ 分析完成！")
    print("Estimated Theta1 parameters:")
    for k, v in theta1.items():
        print(f"  {k:8s} = {v:.6f}")

    print(f"\nEstimated Q_lambda = {Q_lambda_est:.6e}\n")
    print(f"测试集 lambda 路径已保存到: {output_lambda_path}")
    print(f"测试集 RUL 结果已保存到:   {output_rul_path}")

    return theta1, Q_lambda_est, df_test_filtered, df_rul


# ============================================================
# 7. 运行入口
# ============================================================

if __name__ == "__main__":
    train_path = r"C:\Users\lenovo\MRDQC\train.csv"
    test_path = r"C:\Users\lenovo\MRDQC\test.csv"

    theta1, Q_lambda_est, df_test_filtered, df_rul = train_and_predict_rul(
        train_path=train_path,
        test_path=test_path,
        output_rul_path=r"C:\Users\lenovo\MRDQC\rul_predictions.csv",
        output_lambda_path=r"C:\Users\lenovo\MRDQC\test_lambda_paths.csv",
        failure_threshold=0.30,
        R_meas=0.005**2,
    )

    print("\n--- RUL 预测结果 (前20行) ---")
    print(df_rul.head(20))