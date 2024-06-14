import streamlit as st
import time
from requests import Request, Session

api_url = "http://127.0.0.1:6000/phi3"


def response_generator(prompt):
    request = Request(
        'POST', 
        api_url,
        files = {
            'user_input': (None, prompt)
    }
    ).prepare()
    s = Session()
    response = s.send(request)
    output_text = response.json()["message"]

    for word in output_text.split():
        yield word + " "
        time.sleep(0.05)


st.title("LLM ChatBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

