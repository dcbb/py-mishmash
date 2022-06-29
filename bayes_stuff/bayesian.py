import streamlit as st
from scipy import stats
from scipy import optimize
import numpy as np
import altair as alt
import pandas as pd

# st.set_page_config(layout="wide")

def get_posterior_distribution(num_samples, conversion_rate):
    heads = int(num_samples * conversion_rate)
    tails = num_samples - heads
    dist = stats.beta(1+heads, 1+tails)
    return dist, heads, tails

def conversion_rate_difference_posterior(
        rate_a, 
        rate_b, 
        group_size,
        num_difference_samples=10_000
    ):
    conversions_a = stats.binom.rvs(n=group_size, p=rate_a)
    conversions_b = stats.binom.rvs(n=group_size, p=rate_b)
    rate_posterior_a_samples = stats.beta(1 + conversions_a, 1 + group_size - conversions_a).rvs(num_difference_samples)
    rate_posterior_b_samples = stats.beta(1 + conversions_b, 1 + group_size - conversions_b).rvs(num_difference_samples)
    difference_posterior = stats.rv_histogram(np.histogram(rate_posterior_b_samples - rate_posterior_a_samples, bins='auto'))
    q_low, q_high = difference_posterior.ppf(q=[0.05, 0.95])
    return q_low, q_high

def hdi_above_rate(hdi_above,
                   control_conversion_rate,
                   group_size,
                   test_group_delta,
                   iterations=100
                   ):
    detections = 0
    for i in range(iterations):
        ql, qh = conversion_rate_difference_posterior(
            control_conversion_rate, 
            control_conversion_rate + test_group_delta,
            group_size
        )
        if ql > hdi_above:
            detections += 1
    return detections / iterations

func = lambda test_group_delta: 0.95 - hdi_above_rate(test_group_delta=test_group_delta, hdi_above=-0.015, control_conversion_rate=0.1, group_size=1000)
delta, details = optimize.bisect(f=func, a=-0.015, b=0.5, xtol=0.01, full_output=True)
st.write(delta)
rate = hdi_above_rate(test_group_delta=delta, hdi_above=-0.015, control_conversion_rate=0.1, group_size=1000)
st.write(rate)
st.write(f"{details.converged}, {details.function_calls}, {details.iterations}")

st.write("# Explore Bayes stuff")

st.write("## A Beta Binomial model")

st.write("### Observations")

col_a, col_b = st.columns(2)

with col_a:
    num_samples_a = st.slider("number of samples in group A", min_value=0, max_value=1000, value=10)
    conversion_rate_a = st.slider("conversion rate group A", min_value=0.0, max_value=1.0, value=0.1)
    dist_a, heads_a, tails_a = get_posterior_distribution(num_samples_a, conversion_rate_a)
    st.write(f"Group A: {heads_a+tails_a} observations, {heads_a} conversions.")

with col_b:
    num_samples_b = st.slider("number of samples in group B", min_value=0, max_value=1000, value=10)
    conversion_rate_b = st.slider("conversion rate group B", min_value=0.0, max_value=1.0, value=0.1)
    dist_b, heads_b, tails_b = get_posterior_distribution(num_samples_b, conversion_rate_b)
    st.write(f"Group B: {heads_b+tails_b} observations, {heads_b} conversions.")


p = np.linspace(0,1,300)
records = ({'probability': p, 'density': dist.pdf(p), 'group': group} 
    for p in np.linspace(0,1,300)
    for dist, group in [(dist_a, 'A'), (dist_b, 'B')])


st.write("Posteriors of conversion probabilities:")
df_distributions = pd.DataFrame(records)
chart = alt.Chart(df_distributions).mark_line().encode(
    x='probability', y='density', color='group'
)
st.write(chart)


st.write("Posteriors of group differences B – A:")
n = 10_000
dist_dens, edges = np.histogram(dist_b.rvs(n) - dist_a.rvs(n), bins='auto', density=True)
diffs = (edges[1:] + edges[:-1])/2
df_differences = pd.DataFrame({'difference': diffs, 'density': dist_dens})
chart = alt.Chart(df_differences).mark_bar().encode(
    x='difference',
    y='density:Q',
    color=alt.Color('difference:Q', scale=alt.Scale(domainMid=0, scheme='redyellowgreen'))
)
st.write(chart)

st.write("## Power ")

group_size               = st.slider("group size", min_value=100, max_value=10_000, value=100)
control_conversion_rate  = st.slider("control_conversion_rate", min_value=0.01, max_value=0.5, value=0.1)
difference_to_detect_pct = st.slider("difference to detect", min_value=-5.0, max_value=-0.5, value=-1.0, step=0.5)

difference_to_detect = difference_to_detect_pct / 100.0

st.write(f"We want to detect a difference of {difference_to_detect}%.")
st.write(f"Control conversion rate is {control_conversion_rate*100:0.1f}%. Test conversion rate: {(control_conversion_rate+difference_to_detect/100)*100:0.1f}%.")

records = []
for group_diff in [difference_to_detect+off for off in np.linspace(0.0, 0.2, 10+1)]:
    for _ in range(50):
        ql, qh = conversion_rate_difference_posterior(
            control_conversion_rate, 
            control_conversion_rate + group_diff,
            group_size
        )
        records.append({'ql': ql, 'qh': qh, 
                        'hit': ql>difference_to_detect, 
                        'group_diff': group_diff,
                        'true_rate': control_conversion_rate + group_diff, 
                        'row': len(records),
                       })

df = pd.DataFrame(records)
st.write(df.groupby('group_diff').agg({'hit':'mean', 'ql':'mean', 'qh': 'mean'}))
ch = alt.Chart(df, width=800).mark_rule().encode(
    x='row:Q',
    y='ql',
    y2='qh',
    color='group_diff:N',
    row='hit'
)
st.write(ch)

st.write(
    df.groupby('true_rate').agg({'hit': 'mean'})
)

"""
We want the HDI to exclude -1.5%, 90% of the random trials.
What kind uplift do we fail to detect, given sample size?
"""
