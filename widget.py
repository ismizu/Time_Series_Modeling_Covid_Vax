import pickle
from ipywidgets import widgets
import streamlit as st

load_states_dict = open('pickled_data/general_data/states.pickle','rb')
states = pickle.load(load_states_dict)
load_states_dict.close()

selection_box = st.selectbox('State: ',
                              list(states.keys())
                            )

def widget_fig(state_value):
    
    state_value = selection_box
    
    load_graph = open(f'pickled_data/graphs_pickled/{state_value}_graph.pickle','rb')
    fig = pickle.load(load_graph)
    load_graph.close()

    return fig

st.plotly_chart(widget_fig(selection_box))