#%%
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

#%%

df_path = '/home/lag36/scratch/lag36/2025-06-03_lLAG8-10_Merged-Analysis/2025-06-04_lLAG8_ExpNum-Fixed/Steady_State_Combined_df_Estimators.pkl'
df = pd.read_pickle(df_path)
#%%
st.title("Submarlin Stream")
st.write("This is a web app to visualize processed MARLIN results.")


choice = st.radio(
    label="Select subset view:",
    options=['All', 'Long, Normal Growth Rate', 'Long, Slow Growth Rate'],
    index=0
)
if choice == 'All':
    df_show = df
elif choice == 'Long, Normal Growth Rate':

    df_show = (df
        .query('`Length (um)` > 4 and `Growth Rate (1/hr)` > 0.77')
        .sort_values(by='Length (um)', ascending=False)
    )
    
    df = (
        df.assign(color_cat = lambda df_: np.where(
            (df_['Length (um)'] > 4) & (df_['Growth Rate (1/hr)'] > 0.77),
            2,
            df_['color_cat'])
        )
        # .astype({'color_cat': 'category'})
    )
else:
    df_show = df

st.dataframe(df_show)

fig0 = px.scatter(
    df,
    x="Growth Rate (1/hr)",
    y="Length (um)",
    log_y=True,
    color=df['color_cat'].apply(lambda x: 'red' if x == 0 else 'blue' if x == 1 else 'green'),
    color_discrete_map={'red': 'red', 'blue': 'blue', 'green': 'green'},
)

# fig1 = px.scatter(
#     df_show,
#     x="Growth Rate (1/hr)",
#     y="Length (um)",
#     log_y=True,
#     color=df_show['Category'].apply(lambda x: 'red' if x == 'control' else 'blue'),
#     color_discrete_map={'red': 'red', 'blue': 'blue'},
# )

# fig2 = px.scatter(
#     df_show,
#     x="Growth Rate (1/hr)",
#     y="Width (um)",
#     log_y=True,
#     color=df_show['Category'].apply(lambda x: 'red' if x == 'control' else 'blue'),
#     color_discrete_map={'red': 'red', 'blue': 'blue'},
# )

# fig3 = px.scatter(
#     df_show,
#     x="Length (um)",
#     y="Septum Displacement",
#     log_x=True,
#     color=df_show['Category'].apply(lambda x: 'red' if x == 'control' else 'blue'),
#     color_discrete_map={'red': 'red', 'blue': 'blue'},
# )

st.plotly_chart(fig0)
st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)

#%%
