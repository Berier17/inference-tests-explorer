# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 13:39:41 2025

@author: aliel
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


st.set_page_config(page_title="Inferential Statistics ", layout="wide")

st.title("Inferential Statistics — Interactive Summary")

# Sidebar navigation (multi-page alternative)
page = st.sidebar.radio("Go to", [
    "Home",
    "Distributions",
    "Estimation",
    "Hypothesis Tests",
    "ANOVA & Chi-square",
    "Regression",
    "Nonparametrics",
])

# Helper to show an animation if available
def show_animation(name: str, caption: str=""):
    path = f"D:/SamplingDistribution.mp4"
    if os.path.exists(path):
        st.video(path)
        if caption:
            st.caption(caption)
    else:
        st.info("Render animations with: `python manim_scenes/render_all.py` to see videos here.")

def page_home():
    st.write("""
    Welcome! Use the sidebar to explore distributions, estimation, hypothesis testing,
    ANOVA & chi-square, regression, and nonparametrics. Each section shows formulas,
    assumptions, and interactive demos. If you've rendered the Manim animations,
    you'll see short videos embedded inline.
    """)
    st.subheader("Central Limit Theorem (CLT)")
    st.write("The Central Limit Theorem (CLT) states that, regardless of the original population's distribution, the distribution of sample means from that population will approach a normal (bell-shaped) curve as the sample size (n) increases. ")
    st.latex(r"""
    	Z = (X̄ - μ) / (σ/√n).
    """)
    st.markdown("""
                Z ≡ Z-Score\n
                X̄ ≡ Sample Mean\n
                μ ≡ Population Mean\n
                σ ≡ Standard Deviation\n
                n ≡ Sample Size\n
                """)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Sample mean simulation**")
        dist = st.selectbox("Population distribution", ["Uniform(0,1)", "Exponential(λ=1)", "Bernoulli(p=0.5)", "Normal(0,1)"])
        n = st.slider("Sample size n", 1, 500, 30, 1)
        reps = st.slider("Replications", 100, 5000, 1000, 100)
        rng = np.random.default_rng(42)
        if st.button("Simulate means"):
            if dist.startswith("Uniform"):
                data = rng.uniform(size=(reps, n))
            elif dist.startswith("Exponential"):
                data = rng.exponential(scale=1.0, size=(reps, n))
            elif dist.startswith("Bernoulli"):
                data = rng.binomial(1, 0.5, size=(reps, n))
            else:
                data = rng.normal(size=(reps, n))
            means = data.mean(axis=1)

            fig = plt.figure(figsize=(8, 4))
            plt.hist(means, bins=40, density=True, alpha=0.8)
            plt.xlabel("Sample means")
            plt.ylabel("Density")
            st.pyplot(fig, clear_figure=True)

    with col2:
        st.write("**CLT animation**")
        show_animation("clt_convergence", "CLT: standardized mean approaches Normal(0,1).")

    st.divider()
    st.subheader("Reference Distributions — Quick Guide")
    st.latex(r"""\mathcal{N}(\mu,\sigma^2),\quad t_\nu,\quad \chi^2_\nu,\quad F_{\nu_1,\nu_2}""")
    st.write("""
    - Known σ → z-test — when population standard deviation is known, data is normal or n is large.
    - Unknown σ → t-test — when population standard deviation is unknown, data is normal.
    - Variance → Chi-square (χ²) — for tests about a single variance.
    - Ratio of variances → F — for comparing variances (e.g., ANOVA).
    - Proportion → (Wald z‑CI) — For estimating a population proportion 𝑝 based on a sample proportion 𝑝^.
    """)
    
    
    
    
    
def page_distributions():
    st.header("Distributions & Sampling Distributions")

    st.markdown("### Normal vs t")
    nu = st.slider("Degrees of freedom (t_ν)", 1, 100, 10)
    x = np.linspace(-4, 4, 400)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(x, stats.norm.pdf(x), label="Normal(0,1)")
    plt.plot(x, stats.t.pdf(x, df=nu), label=f"t({nu})")
    plt.legend()
    plt.xlabel("x"); plt.ylabel("Density")
    st.pyplot(fig, clear_figure=True)

    st.markdown("### χ² and F")
    x2 = np.linspace(0.001, 30, 400)
    df = st.slider("ν for χ²", 1, 30, 5)
    fig2 = plt.figure(figsize=(8, 4))
    plt.plot(x2, stats.chi2.pdf(x2, df), label=f"chi2({df})")
    plt.legend()
    plt.xlabel("x"); plt.ylabel("Density")
    st.pyplot(fig2, clear_figure=True)

    st.markdown("#### F Distribution")
    x3 = np.linspace(0.001, 5, 400)
    d1 = st.slider("ν₁ (numerator)", 1, 50, 5, key="d1")
    d2 = st.slider("ν₂ (denominator)", 1, 50, 10, key="d2")
    fig3 = plt.figure(figsize=(8, 4))
    plt.plot(x3, stats.f.pdf(x3, d1, d2), label=f"F({d1},{d2})")
    plt.legend(); plt.xlabel("x"); plt.ylabel("Density")
    st.pyplot(fig3, clear_figure=True)

    st.markdown("### CLT — Standard Error")
    st.write("The standard error (SE) is a statistical term that quantifies the accuracy with which a sample distribution represents a population")
    st.latex(r"""\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}} \approx \frac{s}{\sqrt{n}}""")

def page_estimation():
    st.header("Estimation & Confidence Intervals")
    st.write("Estimation is the process of using sample data to calculate a statistic that represents a population parameter, such as a point estimate.")
    st.write("A confidence interval is a range of values that provides a level of certainty that the true, unknown population parameter lies within that range, indicating the uncertainty surrounding the estimate.")

    st.markdown("### Mean, σ known (z-CI)")
    st.latex(r"""\bar{x} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}""")
    col3, col4 = st.columns(2)
    with col3:
        expn = st.expander("When to use?")
        expn.write("""
            - Large sample size (n ≥ 30) OR population variance is known.
            - You want to compare your sample mean to a known benchmark.""")
    
    with col4:
        expn1 = st.expander("Use case?")
        expn1.write("""
            - Marketing: Test if the average ad click-through rate in a campaign is different from the industry benchmark (say 2.5%).

            - Crypto: Test if the mean daily return of Bitcoin differs from 0.1% (a hypothesized "normal" return).

            - Football: Test if a player’s average sprint speed is different from the league average of 30 km/h (where variance data is known).
                    """)

    st.markdown("### Mean, σ unknown (t-CI)")
    st.latex(r"""\bar{x} \pm t_{\alpha/2,\ \nu}\frac{s}{\sqrt{n}},\ \ \nu=n-1""")
    st.write(r"$\bar{x}$ ≡ sample mean")
    st.write(r"$s$ ≡ sample standard deviation")
    st.write(r"$v=n-1$ ≡ degrees of freedom for t-distribution")
    col5, col6 = st.columns(2)
    with col5:
        expn2 = st.expander("When to use?")
        expn2.write("""
            - Population variance is unknown.

            - Small sample size (n < 30).
                    """)
    with col6:
        expn3 = st.expander("Use case?")
        expn3.write("""
            - Marketing: Test if the mean customer satisfaction score from a survey (n = 15) differs from the company’s target score of 8/10.

            - Crypto: Test if the average transaction fee in a small sample of Ethereum blocks differs from $5 (without knowing population variance).

            - Football: Test if the average number of goals scored by a striker in 10 games is significantly higher than his expected average of 0.5 goals per match.
                    """)
                    
    st.markdown("### Proportion (Wald CI)")
    st.latex(r"""\hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}""")
    st.write(r"$\hat{p}$ ≡ sample proportion")
    col7, col8 = st.columns(2)
    with col7:
        expn4 = st.expander("When to use?")
        expn4.write("""
            - Sample size is large (𝑛𝑝^>5np^>5 and 𝑛(1−𝑝^)>5n(1−p^)>5)

            - Proportion isn’t too close to 0 or 1.
                    """)
    with col8:
        expn5 = st.expander("Use case?")
        expn5.write("""
            - Marketing: Estimating the proportion of customers who clicked an ad after seeing it.

            - Crypto: Estimating the proportion of transactions on a blockchain flagged as fraudulent in a given week.

            - Football: Estimating the proportion of penalty kicks successfully scored in a season.
                    """)
    st.write("\n")
    st.write("\n")
    st.markdown("*Although the Wald interval is the most commonly taught, it tends to perform poorly with small samples or when the estimated proportion is very close to 0 or 1. In such cases, the Wilson score interval is preferred because it provides more reliable coverage even for small or extreme proportions, while the Agresti–Coull interval offers a simple adjustment to the Wald method that improves accuracy without being much harder to compute.*")


    st.markdown("#### Interactive CI for μ (σ unknown)")
    n = st.slider("n", 5, 500, 30)
    mu = st.number_input("True μ", value=0.0)
    sigma = st.number_input("True σ", value=1.0)
    alpha = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1)
    reps = st.slider("Replications", 50, 2000, 200)
    rng = np.random.default_rng(0)

    if st.button("Simulate CIs"):
        covered = 0
        intervals = []
        for _ in range(reps):
            x = rng.normal(mu, sigma, size=n)
            xbar = x.mean()
            s = x.std(ddof=1)
            from scipy import stats
            tcrit = stats.t.ppf(1-(1-alpha)/2, df=n-1)
            lo = xbar - tcrit*s/np.sqrt(n)
            hi = xbar + tcrit*s/np.sqrt(n)
            intervals.append((lo, hi))
            if lo <= mu <= hi:
                covered += 1
        rate = covered/reps
        st.write(f"Empirical coverage ≈ {rate:.3f} (target: {alpha})")
        # Plot a subset of intervals
        k = min(50, reps)
        fig = plt.figure(figsize=(8, 4))
        for i in range(k):
            lo, hi = intervals[i]
            plt.plot([lo, hi], [i, i], lw=2)
            plt.plot([mu, mu], [-1, k], linestyle="--")
        plt.xlabel("μ"); plt.ylabel("Interval index")
        st.pyplot(fig, clear_figure=True)

def page_tests():
    st.header("Hypothesis Testing")
   
    st.write(r"""
        Hypothesis testing is a way to use sample data to decide whether there’s enough evidence to support a claim about a population.
        It begins by stating two competing statements: the null hypothesis $H_0$, which represents the status quo or “no effect” assumption, and the alternative hypothesis ($H_1$ or $H_a$), which represents the effect or difference you suspect.
        """)
    st.subheader("One Sample Test")
    st.write("The one-sample t-test is used to check whether the mean of a sample differs significantly from a known or hypothesized population mean.")
    st.markdown("### One-sample z-test (σ known)")
    st.latex(r"""
    Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} 
\ \sim\ \mathcal{N}(0,1) 
\quad \text{under } H_0
    """)
    col9, col10 = st.columns(2)
    with col9:
        expn5 = st.expander("When to use?")
        expn5.write("""
            - You have one sample and want to compare it to a known benchmark.
            - Population standard deviation σ is known (rare in practice).
            - Sample size is reasonably large (n ≥ 30) or the population is normally distributed.
                    """)
    with col10:
        expn6 = st.expander("Use case?")
        expn6.write("""
            - Marketing: Compare average click-through rate to a known industry benchmark.
            - Crypto: Compare average daily return of a cryptocurrency to a historical mean if variance is known.
            - Football: Compare average sprint speed of players to a known league average (σ known).
                    """)

    st.markdown("### One-sample t-test (σ unknown)")
    st.latex(r"""
    T = \frac{\bar{X} - \mu_0}{s / \sqrt{n}} \sim t_{n-1} \quad \text{under } H_0
    """)
    col11, col12 = st.columns(2)
    with col11:
        expn7 = st.expander("When to use?")
        expn7.write("""
            - You have a single sample and want to compare it to a benchmark (expected mean).
            - Population standard deviation σ is unknown.
            - Sample data is roughly symmetric and not heavily skewed.                    """)
    with col12:
        expn8 = st.expander("Use case?")
        expn8.write("""
            - Marketing: Test if the average customer satisfaction score from a survey differs from the company’s target score.
            - Crypto: Test if the mean daily return of a cryptocurrency differs from an expected benchmark return when historical volatility is unknown.
            - Football: Test if a player’s average sprint speed over a season differs from the league average when only a sample of games is available.
                    """)


    st.markdown("### Two-sample t-test (equal variances)")
    st.write("The Two-sample t-test compares the means of two independent groups to see if they are significantly different, assuming both groups have the same variance.")
    st.latex(r"""
    T = \frac{\bar{X}_1 - \bar{X}_2}
{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
\quad\text{where}\quad
s_p^2 = \frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2},
\quad
T \sim t_{n_1 + n_2 - 2}
    """)
    col13, col14 = st.columns(2)
    with col13:
        expn9 = st.expander("When to use?")
        expn9.write("""
            - Comparing two independent samples.
            - Population variances are assumed equal (can be checked with an F-test or Levene’s test).
            - Population standard deviations are unknown (σ not given).                    
            """)
    with col14:
        expn10 = st.expander("Use case?")
        expn10.write("""
            - Healthcare: Compare average recovery times for patients receiving two different treatments.
            - Education: Test if average exam scores differ between two teaching methods.
            - Sports: Compare the average goals scored by two different teams across a season.
                    """)


    st.markdown("### Proportion z-test")
    st.write("The Proportion z-test checks whether the proportions of “successes” are different in two independent groups.")
    st.latex(r"""
    Z = \frac{\hat{p} - p_0}
{\sqrt{\frac{p_0(1 - p_0)}{n}}}
\ \sim\ \mathcal{N}(0,1) 
\quad \text{under } H_0
    """)
    st.write(r"$\hat{p}$ ≡ sample proportion")
    st.write(r"${p}_0$ ≡ null hypothesis proportion")
    
    col15, col16 = st.columns(2)
    with col15:
        expn11 = st.expander("When to use?")
        expn11.write("""
            - Outcome is binary (success/failure, yes/no).
            - Comparing two independent proportions.
            - Sample sizes are large enough for the normal approximation to hold.                    
            """)
    with col16:
        expn12 = st.expander("Use case?")
        expn12.write("""
            - Marketing: Compare the proportion of users who clicked an ad between two campaign designs.
            - E-commerce: Test if the conversion rate differs between two website layouts (A/B testing).
            - Elections: Compare proportions of voters favoring two candidates across different regions.
                    """)

    

    st.markdown("#### p-value & power (approximate)")
    from scipy import stats
    alpha = st.selectbox("α (significance)", [0.01, 0.05, 0.1], index=1)
    mu0 = st.number_input("Null mean μ₀", value=0.0)
    muA = st.number_input("True mean under H₁", value=0.5)
    sigma = st.number_input("σ (known for z)", value=1.0)
    n = st.slider("n", 5, 1000, 50)

    # Two-sided z-test approximate power
    zcrit = stats.norm.ppf(1-alpha/2)
    se = sigma/np.sqrt(n)
    noncen = abs(muA-mu0)/se
    power = stats.norm.sf(zcrit-noncen) + stats.norm.cdf(-zcrit-noncen)
    st.write(f"Approx. two-sided z-test power: **{power:.3f}**")

    # Visualize rejection regions
    x = np.linspace(-4, 4, 400)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(x, stats.norm.pdf(x), label="H0: N(0,1)")
    plt.axvline(zcrit, linestyle="--"); plt.axvline(-zcrit, linestyle="--")
    plt.legend(); plt.xlabel("Z"); plt.ylabel("Density")
    st.pyplot(fig, clear_figure=True)

def page_anova_chi2():
    st.header("ANOVA & Chi-square")
    st.write(r"""
        ANOVA, or Analysis of Variance, is a test used to determine differences between research results from three or more unrelated samples or groups.
        
        A chi-square (Χ²) is a statistical test used to determine if there is a significant association between two categorical variables by comparing observed frequencies to expected frequencies.
             """)

    st.markdown("### One-way ANOVA")
    st.latex(r"""
    F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} \sim F_{k-1,\ N-k}
    """)
    st.write(r"${MS}_{\text{between}}$ ≡ Mean Square Between Groups")
    st.write(r"${MS}_{\text{within}}$ ≡ Mean Square Within Groups")
    st.write(r"$k$ ≡ The Number of Groups")
    st.write(r"$N$ ≡ The Total Number of Observations")
    st.write(r"$F$ ≡ The resulting F-statistic follows an F-distribution with 𝑘−1 and 𝑁−𝑘 degrees of freedom under the null hypothesis")
    st.write("\n")
    st.write("\n")
    
    col17, col18 = st.columns(2)
    with col17:
        expn13 = st.expander("When to use?")
        expn13.write("""
            - Comparing means across 3 or more groups.
            - Independent samples.
            - Populations are normally distributed and have equal variances.                    
            """)
    with col18:
        expn14 = st.expander("Use case?")
        expn14.write("""
            - Agriculture: Compare average crop yield under three different fertilizers.
            - Healthcare: Compare blood pressure levels across patients on three different diets.
            - Business: Test if average customer satisfaction differs across multiple store locations.
                    """)
    
    
    st.write("Simulate groups and compute F-statistic:")
    k = st.slider("Groups (k)", 2, 6, 3)
    n = st.slider("Per-group n", 3, 200, 20)
    mu_diff = st.slider("Mean difference between groups", 0.0, 2.0, 0.0, 0.1)
    sigma = st.number_input("σ (within)", value=1.0)
    rng = np.random.default_rng(7)

    if st.button("Simulate ANOVA"):
        data = []
        for i in range(k):
            mu = i*mu_diff
            data.append(rng.normal(mu, sigma, size=n))
        x = np.concatenate(data)
        grand_mean = x.mean()
        ss_between = sum(n*(xg.mean()-grand_mean)**2 for xg in data)
        ss_within = sum(((xg - xg.mean())**2).sum() for xg in data)
        ms_between = ss_between/(k-1)
        ms_within  = ss_within/(k*(n-1))
        F = ms_between/ms_within
        df1, df2 = k-1, k*(n-1)
        from scipy import stats
        p = 1 - stats.f.cdf(F, df1, df2)
        st.write(f"F = {F:.3f}, df1={df1}, df2={df2}, p-value = {p:.4f}")

        fig = plt.figure()
        for i, xg in enumerate(data):
            xs = np.random.normal(i+1, 0.03, size=n)
            plt.plot(xs, xg, "o", alpha=0.7)
        plt.xlabel("Group"); plt.ylabel("Value")
        st.pyplot(fig, clear_figure=True)

    st.markdown("### Chi-square tests")
    st.write("A chi-squared (χ²) test is a statistical hypothesis test used with categorical data (data in categories, not numerical measurements) to compare observed frequencies against expected frequencies")
    exp5 =st.expander("Types?")
    exp5.write("""
        - Chi-square goodness-of-fit test — Compares the observed distribution of a single categorical variable to an expected distribution.
        - Chi-square test of independence (association) — Determines whether there is a significant relationship between two categorical variables
""")
    st.latex(r"""\chi^2 = \sum \frac{(O-E)^2}{E} \sim \chi^2_{\text{df}}""")
    col19, col20 = st.columns(2)
    with col19:
        expn15 = st.expander("When to use?")
        expn15.write("""
            - Data is categorical (counts/frequencies).
            - Large enough sample size (expected counts ≥ 5 in most cells).
            - Used to test either distribution fit or independence between two categorical variables.                    
            """)
    with col20:
        expn16 = st.expander("Use case?")
        expn16.write("""
            - Chi-Square Test: Check if actual customer responses (click/ignore/unsubscribe) differ from the expected campaign proportions.
            - Independence: Test if gender and voting preference are independent in an election survey.
            - Retail: Test if product category (e.g., electronics, clothing, groceries) is independent of customer region.
                    """)
    st.write("\n")
    st.write("Goodness-of-fit example (k categories):")
    k2 = st.slider("k categories", 2, 8, 4, key="kcat")
    obs = np.array([st.number_input(f"O[{i+1}]", value=10.0, key=f"obs{i}") for i in range(k2)])
    probs = np.array([st.number_input(f"π[{i+1}]", value=1.0/k2, key=f"pi{i}") for i in range(k2)])
    probs = probs / probs.sum()
    n_total = obs.sum()
    E = n_total * probs
    chi2 = ((obs-E)**2/E).sum()
    df = k2-1
    from scipy.stats import chi2 as chi2dist
    p = 1-chi2dist.cdf(chi2, df)
    st.write(f"χ² = {chi2:.3f}, df = {df}, p = {p:.4f}")

def page_regression():
    st.header("Regression Inference")
    st.write("Regression inference is about using sample data to make statements about the true relationship between an independent variable 𝑥 and a dependent variable 𝑦")
    st.video(f"D:Perspective4D.mp4")
    st.markdown("### Model")
    st.latex(r"""y_i = \beta_0 + \beta_1 x_i + \varepsilon_i,\quad \varepsilon_i \sim \mathcal{N}(0,\sigma^2)""")
    st.write("\n")
    st.write(r"$\beta_0$ ≡ the intercept (where the line crosses the y-axis)")
    st.write(r"$\beta_1$ ≡ the slope (how much 𝑦 changes for each unit of 𝑥)")
    st.write(r"$\varepsilon_i$ ≡ the random error, assumed to follow a normal distribution")

    st.markdown("### Inference on β₁")
    st.latex(r"""T = \frac{\hat{\beta}_1 - 0}{\text{SE}(\hat{\beta}_1)} \sim t_{n-2}""")
    st.write(r"$\hat{\beta}_1$ ≡ the estimated slope from the sample regression line")
    st.write(r"$\text{SE}(\hat{\beta}_1)$ ≡ the standard error of the slope estimate")
    st.write(r"$t_{n-2}$ ≡ the t-distribution with 𝑛−2 degrees of freedom")
    
    st.write("Simulate data and compute OLS estimates:")
    n = st.slider("n", 5, 1000, 50)
    beta0 = st.number_input("β₀", value=0.0)
    beta1 = st.number_input("β₁", value=1.0)
    sigma = st.number_input("σ", value=1.0)
    rng = np.random.default_rng(3)
    if st.button("Simulate regression"):
        x = rng.uniform(-2, 2, size=n)
        y = beta0 + beta1*x + rng.normal(0, sigma, size=n)
        X = np.c_[np.ones(n), x]
        beta_hat = np.linalg.inv(X.T@X)@X.T@y
        yhat = X@beta_hat
        resid = y - yhat
        s2 = (resid@resid)/(n-2)
        se_b1 = np.sqrt(s2/((x - x.mean())**2).sum())
        from scipy import stats
        t_stat = beta_hat[1]/se_b1
        p = 2*stats.t.sf(abs(t_stat), df=n-2)
        st.write(f"β̂ = [{beta_hat[0]:.3f}, {beta_hat[1]:.3f}], t for β₁: {t_stat:.3f}, p={p:.4f}")
        fig = plt.figure()
        plt.plot(x, y, "o", alpha=0.7)
        xs = np.linspace(x.min(), x.max(), 200)
        plt.plot(xs, beta_hat[0]+beta_hat[1]*xs)
        st.pyplot(fig, clear_figure=True)

def page_nonparam():
    st.header("Nonparametric Tests (Quick Overview)")
    st.write("Nonparametric tests are hypothesis tests that don't assume your data follows a specific probability distribution, like the normal distribution. They are used when assumptions for parametric tests (like the t-test or ANOVA) aren't met, especially with small samples, categorical or ordinal data, or data with outliers.")

    st.markdown("### Sign Test (one-sample median)")
    st.write("The one-sample sign test is a nonparametric statistical test used to determine if the median of a single sample is different from a hypothesized median.")
    st.latex(r"""
   P(X = x) = \binom{n}{x} (0.5)^x (0.5)^{n - x} = \binom{n}{x} (0.5)^n
    """)
    st.write(r"$x$ ≡ Number of positive signs (i.e., sample values greater than the hypothesized median $𝑚_0$).")
    st.write(r"$n$ ≡ Total number of non-zero differences (values not equal to $𝑚_0$).")

    
    col21, col22 = st.columns(2)
    with col21:
        exp = st.expander("When to use?")
        exp.write("""

           - The data is not normally distributed
           - The sample size is small
           - The data includes outliers that would distort parametric tests like the t-test.
           """)
    with col22:
        exp2 = st.expander("Use case?")
        exp2.write("""
            - Fitness: Check if more athletes improved than declined after a new workout plan.
            - Business: Test if more employees rated the new HR policy as “better” than “worse.”
            - Healthcare: Check if more patients’ symptoms improved after a treatment.
                    """)


    st.markdown("### Wilcoxon Signed-Rank (paired)")
    st.write("Wilcoxon signed-rank test is a non-parametric statistical test used to compare two paired or dependent samples, such as before-and-after measurements on the same individuals")
    st.latex(r"""
    W = \sum_{i=1}^{n} \text{sign}(d_i) \cdot R_i
    """)
    st.write(r"$d_i$ ≡ The difference between paired observations (e.g., before vs. after).")
    st.write(r"$sign(d_i)$ ≡ Indicates whether the difference is positive or negative.")
    st.write(r"$R_i$ ≡ The rank of the absolute difference ∣𝑑𝑖∣, ignoring zero differences.")


    col23, col24 = st.columns(2)
    with col23:
        exp3 = st.expander("When to use?")
        exp3.write("""
           - Paired observations (e.g., same person measured twice).
           - Data is ordinal or continuous but not normally distributed.
           - Looking at differences between conditions.""")
    with col24:
        exp4 = st.expander("Use case?")
        exp4.write("""
            - Finance: Compare stock prices before and after an earnings announcement.
            - Education: Compare students’ test scores before and after a training program.
            - Healthcare: Compare blood pressure of patients before and after taking medication.
                    """)
   
    

    st.markdown("### Mann–Whitney U (two-sample)")
    st.write("""Mann–Whitney U test is a non-parametric test used to compare two independent samples to determine if there is a statistically significant difference between them, particularly when the data is not normally distributed. Also known as the Wilcoxon rank-sum test.""")
    
    st.latex(r"""
   U = n_1 n_2 + \frac{n_1(n_1 + 1)}{2} - R_1
    """)
    st.write(r"$n_1,n_2$ ≡ Sample sizes of the two independent groups.")
    st.write(r"$R_1$ ≡ Sum of the ranks for group 1 (after pooling and ranking all observations).")

    st.write("\n")
    st.write("*These tests avoid strict normality assumptions and are robust to outliers.*")
    col25, col26 = st.columns(2)
    with col25:
        exp5 = st.expander("When to use?")
        exp5.write("""
           - Comparing two independent groups.
           - Data is ordinal or continuous but not normally distributed.
           - Small sample sizes or presence of outliers.""")
    with col26:
        exp6 = st.expander("Use case?")
        exp6.write("""
            - Healthcare: Compare recovery times between patients treated with two different drugs.
            - E-commerce: Compare order values between male and female customers.
            - Sports: Compare reaction times of athletes from two teams.
                    """)
    
    st.markdown('### Kruskal–Wallis H Test')
    st.write("The Kruskal–Wallis H test is a non-parametric statistical method used to compare more than two independent groups when the data doesn't meet the assumptions of a parametric test, such as ANOVA")
    st.latex(r"""
H = \frac{12}{N(N+1)} \sum_{i=1}^k \frac{R_i^2}{n_i} - 3(N+1)
""")
    st.write(r"$N$ ≡ Total number of observations across all groups.")
    st.write(r"$K$ ≡ Number of groups")
    st.write(r"$n_i$ ≡ Sample size of group 𝑖.")
    st.write(r"$R_i$ ≡ Sum of ranks for group 𝑖, after ranking all observations together.")


    col27, col28 = st.columns(2)
    with col27:
        exp7 = st.expander("When to use?")
        exp7.write("""
           - More than two independent groups.
           - Ordinal or continuous non-normal data.
           - Testing overall group differences.""")
    with col28:
        exp8 = st.expander("Use case?")
        exp8.write("""
            - Marketing: Compare click-through rates of three ad campaigns.
            - Crypto: Compare transaction fees across multiple blockchains.
            - Sports: Compare performance scores across three different leagues.
                    """)    
 
    
    st.markdown("### Spearman’s Rank Correlation (𝜌)")
    st.write("Spearman's Rank Correlation Coefficient is a non-parametric statistical measure that assesses the strength and direction of a monotonic relationship between two ranked variables.")
    st.latex(r"r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}")
    st.write(r"$d_i$ ≡ The difference between the ranks of the 𝑖-th pair of observations.")
    st.write(r"$n$ ≡ Number of paired observations.")
    
    
    col29, col30 = st.columns(2)
    with col29:
        exp9 = st.expander("When to use?")
        exp9.write("""
           - Variables are ordinal or not normally distributed.
           - Relationship may not be linear but monotonic.
           - Looking for correlation without strict parametric assumptions.""")
    with col30:
        exp10 = st.expander("Use case?")
        exp10.write("""
            - Marketing: Correlation between ad spend rank and sales rank.
            - Crypto: Correlation between coin market cap rank and trading volume rank.
            - Football: Correlation between player fitness rank and minutes played.
                    """)    


# Router
if page == "Home":
    page_home()
elif page == "Distributions":
    page_distributions()
elif page == "Estimation":
    page_estimation()
elif page == "Hypothesis Tests":
    page_tests()
elif page == "ANOVA & Chi-square":
    page_anova_chi2()
elif page == "Regression":
    page_regression()
else:
    page_nonparam()
