#%%
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(
    # layout="wide",
    page_title="Submarlin Stream",
    page_icon="🛳️",
)
#%%

@st.cache_data(show_spinner=False)
def load_df(path):
    df_ = pd.read_pickle(path)
    df_ = df_.assign(plot_category=np.where(df_['Category'] == 'control', 'control', ''))
    return df_

df_path = '/home/lag36/scratch/lag36/2025-06-03_lLAG8-10_Merged-Analysis/2025-06-04_lLAG8_ExpNum-Fixed/Steady_State_Combined_df_Estimators.pkl'
df = load_df(df_path)
#%%

st.title("Submarlin Stream")
st.write("This is a web app to visualize processed MARLIN results.")

choice = st.radio(
    label="Select subset view:",
    options=['All',
             'Long, Normal Growth Rate',
             'Long, Slow Growth Rate',
             'Wide',
             'Division Position Shifted'],
    index=0,
    key = 'radio_data-subset-choice'
)

from plotly.subplots import make_subplots

panels = [
    ("Growth Rate (1/hr)", "Length (um)", "Growth Rate vs Length"),
    ("Growth Rate (1/hr)", "Width (um)", "Growth Rate vs Width"),
    ("Length (um)", "Septum Displacement", "Length vs Septum Displacement"),
]

color_map = {"": "white", "control": "red"}

sort_by = {
    'All': {'by': None, 'ascending': True},
    'Long, Normal Growth Rate': {'by': 'Length (um)', 'ascending': False},
    'Long, Slow Growth Rate': {'by': 'Length (um)', 'ascending': False},
}

bounds = {
    'Long, Normal Growth Rate': 
        {'Length (um)': (4, None), 'Growth Rate (1/hr)': (0.77, None)},
    'Long, Slow Growth Rate': 
        {'Length (um)': (4, None), 'Growth Rate (1/hr)': (None, 0.77)},
}

sel_bounds = bounds.get(choice, {})

# --- Compute highlight mask ONCE ---
if sel_bounds:
    mask_highlight = np.ones(len(df), dtype=bool)
    for col, (low, high) in sel_bounds.items():
        if low  is not None: mask_highlight &= df[col].to_numpy() >= low
        if high is not None: mask_highlight &= df[col].to_numpy() <= high
else:
    mask_highlight = np.zeros(len(df), dtype=bool)  # nothing highlighted

# (Optional) sort/filter for table; keep table light
df_show = df[mask_highlight] if sel_bounds else df
# st.dataframe(df_show)

if sort_by[choice]['by'] is not None:
    df_show = df_show.sort_values(
        by=sort_by[choice]['by'],
        ascending=sort_by[choice]['ascending']
    ).reset_index(drop=True)
st.dataframe(df_show)

fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06)

for i, (x_col, y_col, _) in enumerate(panels):
    # --- Base scatter (all points, colored by category) ---
    tmp = px.scatter(
        df,
        x=x_col, y=y_col,
        color="plot_category",
        color_discrete_map=color_map,
        hover_data={
            "opLAG1_id": True, "Gene": True, "N Observations": True,
            x_col: False, y_col: False, "Category": False
        },
        render_mode="webgl",
    )
    for tr in tmp.data:
        tr.update(
            legendgroup=tr.name,
            showlegend=(i == 0),
            marker=dict(size=4, opacity=0.6),
            hoverinfo="skip",
            hovertemplate=None,
        )
        fig.add_trace(tr, row=1, col=i+1)

    # --- Add vertical/horizontal lines if bounds exist ---
    if x_col in sel_bounds:
        xmin, xmax = sel_bounds[x_col]
        if xmin is not None:
            fig.add_vline(x=xmin, line_dash="dash", line_color="green", row=1, col=i+1)
        if xmax is not None:
            fig.add_vline(x=xmax, line_dash="dash", line_color="green", row=1, col=i+1)

    if y_col in sel_bounds:
        ymin, ymax = sel_bounds[y_col]
        if ymin is not None:
            fig.add_hline(y=ymin, line_dash="dash", line_color="green", row=1, col=i+1)
        if ymax is not None:
            fig.add_hline(y=ymax, line_dash="dash", line_color="green", row=1, col=i+1)

    # --- Highlight points inside bounds ---
    if sel_bounds:
        mask = np.ones(len(df), dtype=bool)
        for col, (low, high) in sel_bounds.items():
            if low is not None:
                mask &= df[col] >= low
            if high is not None:
                mask &= df[col] <= high
        highlight = df[mask]

        fig.add_trace(
            go.Scatter(
                x=highlight[x_col],
                y=highlight[y_col],
                mode="markers",
                marker=dict(color="green", size=6, opacity=0.9),
                name="In bounds",
                legendgroup="highlight",
                showlegend=(i == 0),  # single legend entry
                hovertext=highlight["Gene"],
                hoverinfo="text",
            ),
            row=1, col=i+1
        )

    # Axes
    fig.update_xaxes(title_text=x_col, row=1, col=i+1)
    fig.update_yaxes(title_text=y_col, row=1, col=i+1)

fig.update_layout(
    legend_title=None,
    height=360,
    margin=dict(l=30, r=20, t=60, b=40),
    legend=dict(itemclick=False, itemdoubleclick=False),
)
st.plotly_chart(fig)

# st.dataframe(df)

## MINE
# fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06)

# for i, (x_col, y_col, _) in enumerate(panels):
#     tmp = px.scatter(
#         df,
#         x=x_col,
#         y=y_col,
#         color="plot_category",
#         color_discrete_map=color_map,
#         hover_data={
#             "opLAG1_id": True, "Gene": True, "N Observations": True,
#             x_col: False, y_col: False, "Category": False
#         },
#         log_y = True if y_col == "Length (um)" or y_col == "Width (um)" else False,
#     )

#     for tr in tmp.data:
#         tr.update(
#             legendgroup=tr.name,
#             showlegend=(i == 0),
#             marker=dict(size=4, opacity=0.8),
#         )
#         fig.add_trace(tr, row=1, col=i+1)
#     fig.update_xaxes(title_text=x_col,row=1, col=i+1, type="log" if x_col == "Length (um)" or x_col == 'Septum_Displacement' else "linear")
#     fig.update_yaxes(title_text=y_col, row=1, col=i+1, type="log" if y_col == "Length (um)" or y_col == 'Septum Displacement' else "linear")

# fig.update_traces(hovertemplate="gRNA ID: %{customdata[0]}<br>Gene: %{customdata[1]}<br>N Observations: %{customdata[2]}<extra></extra>")
# fig.update_layout(legend_title=None, height=360, margin=dict(l=30, r=20, t=60, b=40))

# st.plotly_chart(fig)
# st.dataframe(df)


# bounds = {
#     'Long, Normal Growth Rate': 
#         {'Length (um)': (4, None), 'Growth Rate (1/hr)': (0.77, None)},
#     'Long, Slow Growth Rate': 
#         {'Length (um)': (4, None), 'Growth Rate (1/hr)': (None, 0.77)},
# }

# if choice == 'Long, Normal Growth Rate':
#     # TODO
# elif choice == 'Long, Slow Growth Rate':
#     # TODO




#%%

# if choice == 'All':

#     df_show = df

#     fig_0 = px.scatter(
#         df,
#         x="Growth Rate (1/hr)",
#         y="Length (um)",
#         hover_data = 'Gene',
#         # text = 'Gene',
#         log_y=True,
#         color=df['color_cat'].apply(lambda x: 'red' if x == 0 else 'blue' if x == 1 else 'green'),
#         color_discrete_map={'red': 'red', 'blue': 'blue', 'green': 'green'},
#     )

# elif choice == 'Long, Normal Growth Rate':
#     length_lower = 4
#     length_upper = None
#     growth_rate_lower = 0.77
#     growth_rate_upper = None

#     df_show = (df
#         .loc[lambda df_: (df_['Length (um)'] > length_lower) & (df_['Growth Rate (1/hr)'] > growth_rate_lower)]
#         .sort_values(by='Length (um)', ascending=False)
#     )

#     df = (
#         df.assign(color_cat = lambda df_: np.where(
#             (df_['Length (um)'] > length_lower) & (df_['Growth Rate (1/hr)'] > growth_rate_lower),
#             2,
#             df_['color_cat'])
#         )
#     )
#     fig_0 = px.scatter(
#         df,
#         x="Growth Rate (1/hr)",
#         y="Length (um)",
#         log_y=True,
#         color=df['color_cat'].apply(lambda x: 'red' if x == 0 else 'blue' if x == 1 else 'green'),
#         color_discrete_map={'red': 'red', 'blue': 'blue', 'green': 'green'},
#     )
#     fig_0.add_hline(y=length_lower, line_dash="dot", line_color="green")
#     fig_0.add_vline(x=growth_rate_lower, line_dash="dot", line_color="green")
#     fig_0.update_traces(marker=dict(size=5))

# else:
#     df_show = df

# st.dataframe(df_show)

# fig_1 = px.scatter(
#     df_show,
#     x="Growth Rate (1/hr)",
#     y="Length (um)",
#     log_y=True,
#     color=df_show['Category'].apply(lambda x: 'red' if x == 'control' else 'blue'),
#     color_discrete_map={'red': 'red', 'blue': 'blue'},
# )

# fig_2 = px.scatter(
#     df_show,
#     x="Growth Rate (1/hr)",
#     y="Width (um)",
#     log_y=True,
#     color=df_show['Category'].apply(lambda x: 'red' if x == 'control' else 'blue'),
#     color_discrete_map={'red': 'red', 'blue': 'blue'},
# )

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# fig = make_subplots(rows=1, cols=3)
# for trace in fig_0['data']:
#     trace.showlegend = False  # Show legend for the first trace
#     fig.add_trace(trace, row=1, col=1)
# for trace in fig_1['data']:
#     fig.add_trace(trace, row=1, col=2)
#     trace.showlegend = False  # Hide legend for subsequent traces
# for trace in fig_2['data']:
#     fig.add_trace(trace, row=1, col=3)
#     trace.showlegend = True  # Hide legend for subsequent traces


# # st.header("AgGrid View of Data")
# # AgGrid(df_show, key=str(uuid.uuid4()))


# # fig0 = px.scatter(
# #     df,
# #     x="Growth Rate (1/hr)",
# #     y="Length (um)",
# #     log_y=True,
# #     color=df['color_cat'].apply(lambda x: 'red' if x == 0 else 'blue' if x == 1 else 'green'),
# #     color_discrete_map={'red': 'red', 'blue': 'blue', 'green': 'green'},
# # )

# # fig1 = px.scatter(
# #     df_show,
# #     x="Growth Rate (1/hr)",
# #     y="Length (um)",
# #     log_y=True,
# #     color=df_show['Category'].apply(lambda x: 'red' if x == 'control' else 'blue'),
# #     color_discrete_map={'red': 'red', 'blue': 'blue'},
# # )

# # fig2 = px.scatter(
# #     df_show,
# #     x="Growth Rate (1/hr)",
# #     y="Width (um)",
# #     log_y=True,
# #     color=df_show['Category'].apply(lambda x: 'red' if x == 'control' else 'blue'),
# #     color_discrete_map={'red': 'red', 'blue': 'blue'},
# # )

# # fig3 = px.scatter(
# #     df_show,
# #     x="Length (um)",
# #     y="Septum Displacement",
# #     log_x=True,
# #     color=df_show['Category'].apply(lambda x: 'red' if x == 'control' else 'blue'),
# #     color_discrete_map={'red': 'red', 'blue': 'blue'},
# # )

# st.plotly_chart(fig, use_container_width=True)
# # st.plotly_chart(fig1)
# # st.plotly_chart(fig2)
# # st.plotly_chart(fig3)

# #%%
