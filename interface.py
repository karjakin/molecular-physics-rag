import ollama
import streamlit as st
from llamaraptor import model_res_generator
st.title("Física Molecular")

# initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿qué es un mol?"):
    # add latest message to history in format {role, content}
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        print(prompt)
        message = st.write_stream(model_res_generator(prompt))
        st.session_state["messages"].append({"role": "assistant", "content": message})
