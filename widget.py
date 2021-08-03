import pickle
import base64
import streamlit as st

st.set_page_config(layout = 'wide', initial_sidebar_state = 'collapsed')

st.title('COVID-19: Vaccination Rate Simulator')
st.markdown('> ### Select a state from the dropdown menu to begin.')
st.sidebar.markdown('''The following models require additional tuning:
- Hawaii
- Maine
- Minnesota
- New Mexico
- Ohio
- Oregon
- South Dakota
- Virginia
- Wisconsin
''')
sidebar_problem = st.sidebar.beta_expander('What\'s wrong?')
sidebar_problem.markdown('''Currently, the models do not accurately alter hospitalizations \
and fatalities as the vaccination rate changes.

#### Why does this happen?

Facebook's Prophet utilizes two powerful effects to increase forecast accuracy. They are:

1. Changepoint Range

2. Changepoint Prior Scale

A changepoint is when Prophet identifies a change in the trend over time.

Changepoint range indicates how far into the data you wish to account for changepoints.

Changepoint prior scale indicates how sensitive you want the model to be when looking for changepoints.

The predictions utilize vaccination rates to predict the target variables (hospitalizations and fatalities). \
Thus, if the changepoint range and sensitivity detect a changepoint during an upward trend, \
the trend will be set to an increase in the target variables. What the model will then believe is \
that the increasing vaccination rate is to blame.''')

sidebar_problem_fix = st.sidebar.beta_expander('What can be done?')
sidebar_problem_fix.markdown('''There are some fixes that are currently in the works:

1. Adjust the changepoint variables:
>- Adjustments can be made to help the model capture the general trend. \
As opposed to capturing many smaller trends.

2. Large increases in hospitalizations and fatalities can be linked to specific events \
(Such as the emergence of the delta strain)
>- The dates of these events can be entered into Facebook Prophet to help it \
identify the overall trend by understanding where anomalies are.''')

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
      background-image: url("data:image/png;base64,%s");
      background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('images/blank.png')

st.markdown(
        """
<style>
    .reportview-container .main .block-container{
        max-width: 1440px;
    }
</style>
""",
        unsafe_allow_html=True,
    )

load_states_dict = open('pickled_data/general_data/states.pickle','rb')
states = pickle.load(load_states_dict)
load_states_dict.close()

selection_box = st.selectbox('State: ', list(states.values()))

def widget_fig(state_value):
    
    state_value = [x for x,y in list(zip(states.keys(), states.values())) if y == selection_box][0]
    
    load_graph = open(f'pickled_data/graphs_pickled/{state_value}_graph.pickle','rb')
    fig = pickle.load(load_graph)
    load_graph.close()

    return fig

st.plotly_chart(widget_fig(selection_box), use_container_width = True)