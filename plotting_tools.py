import altair as alt
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import altair as alt

def build_mh_charts(
    chain_df, 
    i, 
    p_true, 
    n_steps,
    burn_in=0, 
    std_dev=0.05, 
    animate_dist=False,
    dist_scale=0.5
):
    """
    Builds a chart (or charts) for:
      1) Histogram of accepted p-values (left),
      2) MCMC trace (right),
      3) A normal distribution "laid on its side" at the current iteration i 
         if animate_dist=True (overlaid on the trace).

    Args:
        chain_df (pd.DataFrame): M-H chain with columns 
            [iteration, state, candidate, accepted].
        i (int): Current iteration to display.
        p_true (float): True p-value for reference.
        n_steps (int): Total M-H steps (for x-axis domain).
        burn_in (int): Number of burn-in iterations to exclude.
        std_dev (float): Proposal distribution std. dev.
        animate_dist (bool): Whether to show the proposal distribution 
                             around the current state at iteration i.
        dist_scale (float): Factor to scale the pdf horizontally 
                            so it doesn't blow out the chart near iteration i.

    Returns:
        alt.Chart: A horizontal concatenation of 
                   (1) the histogram and 
                   (2) the layered trace + optional distribution.
    """
    # Filter data for burn-in and up to iteration i so the line grows over time
    filtered_df = chain_df[
        (chain_df['iteration'] > burn_in) & (chain_df['iteration'] <= i)
    ].copy()
    
    # Histogram of accepted proposal states
    accepted_df = filtered_df[filtered_df['accepted']]
    hist = alt.Chart(accepted_df).mark_bar(
        orient="horizontal",
        size=10,               # Thicker bars
        stroke='black',        # Outline color
        strokeWidth=1          # Outline thickness
    ).encode(
        y=alt.Y(
            'state:Q',
            bin=alt.Bin(maxbins=20, extent=[0, 1]),
            title='p',
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(values=[x/10 for x in range(11)], labelOverlap=False)
        ),
        x=alt.X('count()', title='Count')
    ).properties(width=200, height=300, title="Histogram of Accepted p-values")

    # Red rule for the true p-value
    rule_df = pd.DataFrame({'p_true': [p_true]})
    true_p_rule = alt.Chart(rule_df).mark_rule(color='red').encode(
        y=alt.Y('p_true:Q', scale=alt.Scale(domain=[0, 1]))
    )
    hist_with_rule = hist + true_p_rule
    
    # Chart trace
    fixed_x_domain = [burn_in + 1, n_steps]
    
    accepted_pts = filtered_df[filtered_df['accepted']]
    rejected_pts = filtered_df[~filtered_df['accepted']]
    
    # Blue line for accepted states
    line_accepted = alt.Chart(accepted_pts).mark_line(color='blue').encode(
        x=alt.X("iteration:Q", title="Iteration", scale=alt.Scale(domain=fixed_x_domain)),
        y=alt.Y("state:Q", title="p", scale=alt.Scale(domain=[0, 1]))
    )
    accepted_chart = alt.Chart(accepted_pts).mark_circle(size=60, color='blue').encode(
        x=alt.X("iteration:Q", scale=alt.Scale(domain=fixed_x_domain)),
        y=alt.Y("state:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=["iteration", "state"]
    )
    
    # Grey ticks for rejected proposals
    rejected_chart = alt.Chart(rejected_pts).mark_tick(color='grey', thickness=2).encode(
        x=alt.X("iteration:Q", scale=alt.Scale(domain=fixed_x_domain)),
        y=alt.Y("candidate:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=["iteration", "candidate"]
    )
    
    # Highlight the current iteration in green
    current_iter_df = filtered_df[filtered_df['iteration'] == i]
    current_pt_chart = alt.Chart(current_iter_df).mark_circle(size=80, color='green').encode(
        x=alt.X("iteration:Q", scale=alt.Scale(domain=fixed_x_domain)),
        y=alt.Y("state:Q", scale=alt.Scale(domain=[0, 1]))
    )
    
    # Combine these trace layers
    trace_layers = line_accepted + accepted_chart + rejected_chart + current_pt_chart
    
    # Add graph for the proposal distribution which we pick samples from
    # TODO: make it clearer what this is?
    dist_layer = None
    if animate_dist and not current_iter_df.empty:
        current_p = current_iter_df['state'].iloc[0]
        
        # Generate a small range around the current p, clipped to [0, 1].
        lower = max(0, current_p - 6*std_dev)
        upper = min(1, current_p + 6*std_dev)
        if lower < upper:  # ensure we have a valid range
            theta_vals = np.linspace(lower, upper, 100)
            
            # Normal PDF
            pdf_vals = (1.0 / (std_dev * np.sqrt(2*np.pi))) * \
                       np.exp(-0.5 * ((theta_vals - current_p) / std_dev)**2)
            # Scale the PDF horizontally
            pdf_vals *= dist_scale
            
            # x = iteration i + scaled pdf, y = theta_vals
            dist_df = pd.DataFrame({
                'x': i + pdf_vals,
                'theta': theta_vals
            })
            
            # TODO: How do I make this 'hollow'?
            dist_layer = alt.Chart(dist_df).mark_line(color='white', strokeWidth=0.1).encode(
                x=alt.X('x:Q', scale=alt.Scale(domain=fixed_x_domain)),
                y=alt.Y('theta:Q', scale=alt.Scale(domain=[0, 1]))
            )
            
            # Optional text label near the peak
            peak_idx = np.argmax(pdf_vals)
            peak_x = dist_df['x'].iloc[peak_idx]
            peak_theta = dist_df['theta'].iloc[peak_idx]
            label_df = pd.DataFrame({
                'x': [peak_x],
                'theta': [peak_theta],
                'text': [f"N({current_p:.2f}, {std_dev:.2f})"]
            })
            text_layer = alt.Chart(label_df).mark_text(
                align='left',
                dx=5,
                color='orange'
            ).encode(
                x='x:Q',
                y='theta:Q',
                text='text:N'
            )
            
            dist_layer = dist_layer + text_layer

    # combine trace + dist if we built dist_layer
    if dist_layer is not None:
        trace_with_dist = (trace_layers + dist_layer).properties(
            width=400, height=300, 
            title=f"MH Trace (iterations > burn-in) up to iteration {i}"
        )
    else:
        trace_with_dist = trace_layers.properties(
            width=400, height=300, 
            title=f"MH Trace (iterations > burn-in) up to iteration {i}"
        )
    
    # Concatenate for final chart
    final_chart = alt.hconcat(hist_with_rule, trace_with_dist)
    return final_chart
