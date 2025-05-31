# Imports.
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go

# Import the data that we will use for visualization.
qb_td = pd.read_parquet('<YOUR_DATASETS_DIRECTORY>/qb_bootstrap_results.parquet')

#---------------------#
# Streamlit app code. #
#---------------------#

# Set page configuration.
st.set_page_config(layout="wide")

# Set the style.
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }

    .stApp {
        background-color: #f5f7fa;
    }

    .css-1d391kg, .css-1v0mbdj, .st-emotion-cache-1v0mbdj {
        background-color: #dbe4ee !important;
    }

    h1 {
        color: #1f4e79;
    }

    .stButton > button {
        background-color: #007ACC;
        color: white;
        border: none;
        border-radius: 5px;
    }

    .stButton > button:hover {
        background-color: #005f99;
        color: white;
    }

    .stAlert {
        border-left: 5px solid #007ACC;
    }
    </style>

    <h1 style='text-align: center; color: #1f4e79;'>
        NFL QB Passing TD Prop Model Predictions
    </h1>
    <p style='text-align: center; color: #4a4a4a; font-size:18px;'>
        Interactive dashboard for visualizing pass TD predictions.
    </p>
""", unsafe_allow_html=True)

# Page settings.
page = st.sidebar.radio('Pages', ['Predictions', 'About'])

# Here is the bulk of the code. We will have a page for predictions and an About page that explains the project and app.
if page == 'Predictions':

    # Create two columns for the main page.
    left_col, right_col = st.columns([1, 1])

    with left_col:

        # Sidebar filters.
        st.sidebar.title("Filter Options")
        teams = sorted(qb_td['team'].unique())
        seasons = sorted(qb_td['season'].unique())
        weeks = sorted(qb_td['week'].unique())

        selected_team = st.sidebar.selectbox("Select Team", teams)
        selected_season = st.sidebar.radio("Select Season", seasons)
        selected_week = st.sidebar.slider("Select Week", min_value=min(weeks), max_value=max(weeks), value=min(weeks))

        # Filter based on the selected team, season, and week.
        qb_td_select = qb_td[(qb_td['team'] == selected_team) & (qb_td['season'] == selected_season) & (qb_td['week'] == selected_week)]

        # Top: Histogram with bootstrapping results.
        if not qb_td_select.empty:
            row = qb_td_select.iloc[0]
            player_name = row['player_display_name']
            defteam = row['defteam']
            y_boot = np.array(row['y_pred_bootstrap'])
            true_val = int(row['y_true'])
            bins = np.arange(0, 8) - 0.5
            counts, edges = np.histogram(y_boot, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2

            hist = go.Bar(x=centers, y=counts, width=1.0, opacity=0.6, name='Relative Frequency', marker_color='forestgreen')

            obs_height = counts[true_val] if 0 <= true_val < len(counts) else 0
            scatter = go.Scatter(x=[true_val], y=[obs_height + 0.02], mode='markers+text', marker=dict(symbol='star', size=14, color='red'), textposition='top center', name='Observed')

            fig_hist = go.Figure(data=[hist, scatter])
            fig_hist.update_layout(title=f'Pass TD Simulations for {player_name} vs {defteam}, Week {selected_week} of {selected_season}', xaxis_title='Number of TD Passes', yaxis_title='Proportion', margin=dict(t=40), template='plotly_dark')
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning('No data available for the selected filters.')

        # Bottom: Reverse stacked bar with probability masses.
        if not qb_td_select.empty:
            columns = ['p_4plus_td', 'p_3_td', 'p_2_td', 'p_1_td', 'p_0_td']
            labels = ["4+ TDs", "3 TDs", "2 TDs", "1 TDs", "0 TDs"]

            # Warm-to-cool colors to correlate with counts.
            colors = ["#e63946", "#f77f00", "#fcbf49", "#a8dadc", "#457b9d"]

            # QB's probabilities for the week.
            probs = qb_td_select.iloc[0][columns].values

            # Here is the chart and layout code.
            fig = go.Figure()
            for label, probs, color in zip(labels, probs, colors):
                fig.add_trace(go.Bar(x=[''], y=[probs], name=label, marker=dict(color=color)))

            fig.update_layout(barmode='stack', title=f'Cumulative Pass TD Probability for {player_name} vs {defteam}, Week {selected_week} of {selected_season}', yaxis_title='Probability', yaxis=dict(range=[0, 1.01]), margin=dict(t=40), template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning('No data available for the selected filters.')

        # Disclaimer.
        st.markdown("""
        <p style='text-align: center; font-size: 13px; color: gray; margin-top: 40px;'>
            <strong>Disclaimer:</strong> This app is for informational and entertainment purposes only.
            No content here constitutes betting advice. Please gamble responsibly. If you or someone you know is struggling with a gambling problem, help is available.
            Visit ncpgambling.org or call 1-800-GAMBLER (1-800-426-2537) for free, confidential support.
        </p>
        """, unsafe_allow_html=True)
    
    with right_col:
        pass

elif page == 'About':
    st.markdown("""
        **Purpose**
        - This app provides an interface to view predictive modeling results of NFL quarterbacks' passing touchdown performances since 2023.
        - During the season, it displays expected values for a hypothetical $100 bet on each available passing touchdown prop.
        
        **Features:**
        - Interactive filtering of bootstrapping results and passing touchdown count probabilities by team, season, and week.
                
        **FAQs**

        **Q1: Why is there no data available for some games?**  
        A: One of three reasons:
        - No predictions are available for a QB's first game in a season. The model requires at least one full-game performance by a QB in a season to predict an upcoming game. 
        - Two or more QBs on the team each played a nontrivial share of the snaps in the game. Predictions are only available for games where a QB played all or almost all snaps for their team.
        - The team was on bye. After Week 18, only playoff teams will have data available for their postseason games.

        **Q2: Is the model being updated on a regular basis?**  
        A: I am always experimenting with tweaks to the model, primarily around feature selection. A new model will be deployed if it surpasses the performance of the current one.

        **Q3: How did you do this?**  
        A: Please check out the [project on GitHub](https://github.com/bochnerjacob/passline) if you are interesting in learning more about how everything works behind the scenes. All code is open source except for the passing TD lines that feed into expected value calculations, because I am not authorized to redistribute that data.

        **Created by:** Jacob Bochner
    """)

    # Disclaimer.
    st.markdown("""
    <p style='text-align: center; font-size: 13px; color: gray; margin-top: 40px;'>
        <strong>Disclaimer:</strong> This app is for informational and entertainment purposes only.
        No content here constitutes betting advice. Please gamble responsibly. If you or someone you know is struggling with a gambling problem, help is available.
        Visit ncpgambling.org or call 1-800-GAMBLER (1-800-426-2537) for free, confidential support.
    </p>
    """, unsafe_allow_html=True)