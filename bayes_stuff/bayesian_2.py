import streamlit as st
from scipy import stats
from scipy import optimize
import numpy as np
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt

st.write("# More Bayes stuff")

c1, c2 = st.columns(2)
with c1:
    accept_confidence = st.slider("Confidence for accepting",min_value=0.9, max_value=0.99, value=0.99, step=0.01)
with c2:
    accept_left_bound = st.slider("Minimum group difference for accepting",min_value=-0.02, max_value=0.05, value=0.0, step=0.01)

st.write(f"""
**Acceptance criterium** 

> We will accept the test if we are {accept_confidence*100:0.0f}% certain that 
> the difference between test and control is at least {accept_left_bound*100:0.1f}%.
""")

c1, c2 = st.columns(2)
with c1:
    control_conversion_rate = st.slider("Control group conversion rate",min_value=0.01, max_value=0.2, value=0.05, step=0.01)
with c2:
    expected_effect = st.slider("Expected conversion rate uplift",min_value=0.01, max_value=0.1, value=0.02, step=0.01)

st.write(f"""
**Expectations** 

> Our control group has a conversion rate of {control_conversion_rate*100:0.0f}%.
> We expect the test group to have an uplift by {expected_effect*100:0.0f}%, and therfore a conversion rate of
> {(control_conversion_rate + expected_effect)*100:0.0f}%.
""")

# group_size = st.slider('Number of samples per group', min_value=100, max_value=20_000, value=1000, step=100)

def conversion_rate_difference_posterior(
        rate_a, 
        rate_b, 
        group_size,
        ci_lower=0.05,
        ci_upper=0.95,
        num_difference_samples=10_000
    ):
    # HERE is the BIG difference!
    if True:
        conversions_a = stats.binom.rvs(n=int(group_size), p=rate_a)
        conversions_b = stats.binom.rvs(n=int(group_size), p=rate_b)
    else:
        conversions_a = int(group_size * rate_a)
        conversions_b = int(group_size * rate_b)
    rate_posterior_a_samples = stats.beta(1 + conversions_a, 1 + group_size - conversions_a).rvs(num_difference_samples)
    rate_posterior_b_samples = stats.beta(1 + conversions_b, 1 + group_size - conversions_b).rvs(num_difference_samples)
    difference_posterior = stats.rv_histogram(np.histogram(rate_posterior_b_samples - rate_posterior_a_samples, bins='auto'))
    q_low, q_high = difference_posterior.ppf(q=[ci_lower, ci_upper])
    return difference_posterior,q_low, q_high


def f(group_size_, iterations=500):
    accept_count = 0
    for _ in range(iterations):
        diff_posterior, c_low, c_high=conversion_rate_difference_posterior(
            rate_a=control_conversion_rate,
            rate_b=control_conversion_rate + expected_effect,
            ci_lower=1.0 - accept_confidence,
            ci_upper=accept_confidence,
            group_size=group_size_,
        )
        if c_low >= accept_left_bound:
            accept_count +=1
    return accept_count / iterations

with st.expander("Sample size computation: Algorithm details, experts only"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sample_size_search_min_samples = st.number_input("min samples", value=100)
    with c2:
        sample_size_search_max_samples = st.number_input("max samples", value=20_000)
    with c3:
        st.number_input("sample size confidence", value=0.95)
    with c4:
        st.number_input("tolerance", value=50)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        simulation_iterations = st.number_input("simulation iterations", value=500)

if st.button('Compute required sample size...'):
    with st.spinner('Crunching numbers...'):
        min_sample_count, details = optimize.bisect(
            f=lambda group_size_: 0.95 - f(group_size_, iterations=simulation_iterations), 
            a=sample_size_search_min_samples, 
            b=sample_size_search_max_samples, 
            xtol=50, 
            full_output=True)

        st.write(f"We need **{int(min_sample_count)} samples** per group.")
        st.write(f"Optimisation details: {details}")

st.write("# more")

if False:

    num_runs = 100
    num_accept = 0

    fig, ax = plt.subplots()
    p = np.linspace(-0.1, 0.1, 300)
    y = None
    for _ in range(num_runs):
        diff_posterior, c_low, c_high=conversion_rate_difference_posterior(
            rate_a=control_conversion_rate,
            rate_b=control_conversion_rate + expected_effect,
            ci_lower=1.0 - accept_confidence,
            ci_upper=accept_confidence,
            group_size=group_size,
        )
        dens = diff_posterior.pdf(p)
        accept = c_low >= accept_left_bound
        if accept:
            num_accept += 1
        color = 'b' if accept else 'r'
        ax.plot(p, dens, color,alpha=0.1)
        if y is None:
            y = ax.get_ylim()
        #ax.plot([c_low, c_low], y, f'{color}:', alpha=0.1)
    ax.plot([accept_left_bound, accept_left_bound], y, 'k')
    st.write(fig)

    st.write(f"Accept rate: {num_accept / num_runs * 100:0.1f}%")

