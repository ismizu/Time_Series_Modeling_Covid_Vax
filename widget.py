import pickle
import base64
import streamlit as st

st.set_page_config(layout = 'wide', initial_sidebar_state = 'collapsed')

st.title('COVID-19: Vaccination Rate Simulator')
st.markdown('> ### Select a state from the dropdown menu to begin.')
st.sidebar.markdown('''The following models require additional tuning:
- Colorado
- Hawaii
- Maine
- Minnesota
- New Hampshire
- Oregon
- Pennsylvania
- West Virgina
''')
expander = st.sidebar.beta_expander('What\'s wrong and what needs to be tuned?')
expander.markdown('''Currently, the model increases hospitalizations\
 and fatalities when the vaccination rate increases.

#### Why does this happen?

Facebook's Prophet utilizes two powerful effects to increase forecast accuracy. They are:

1. Changepoint Range

2. Changepoint Prior Scale

A changepoint is when Prophet identifies a change in the trend over time.

Changepoint range indicates how far into the data you wish to account for changepoints.

Changepoint prior scale indicates how sensitive you want the model to be when looking for changepoints.

Thus, if the changepoint range ends during an upward trend in hospitalizations or fatalities, \
and the changepoint prior scale is set to a sensitivity to catch this upward trend, \
Prophet will set the trend to increase from now on. Since the predictions use vaccination rates \
to help make predictions, the new trend only sees an increasing COVID case rate as well as the \
increasing vaccination rate. Thus, the model predictions reverse. Sizeably so when the vaccination rate \
is multiplied.''')

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
    .reportview-container .main .block-container{{
        max-width: 1440px;
    }}
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