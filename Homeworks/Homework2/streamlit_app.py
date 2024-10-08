import streamlit as st
from typing import List
from datetime import date
from dotenv import load_dotenv
import os
from openai import OpenAI
from Homework2 import Agent, CONVO, Flight, AgentResponse, FindFlightsResponse, BookFlightResponse, load_flights_dataset, SYSTEM_PROMPT

# Initialize OpenAI client
load_dotenv()

client = OpenAI(base_url=os.getenv("URL"), api_key=os.getenv("KEY"))

# Load flights dataset
flights = load_flights_dataset()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = CONVO
if 'agent' not in st.session_state:
    st.session_state.agent = Agent(conversation=st.session_state.messages, flights=flights, client=client)

st.title("LLM Travel Agent Chatbot")

# Display chat messages
for message in st.session_state.messages[1:]:  # Skip the system prompt
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get agent response
    response = st.session_state.agent.say(user_input)
    print(response)

    # Display agent response
    with st.chat_message("assistant"):
        st.write(response.text)

        # Display additional information based on response type
        if isinstance(response, FindFlightsResponse):
            st.write("Available Flights:")
            for flight_id in response.available_flights:
                st.write(f"- Flight ID: {flight_id}")
        elif isinstance(response, BookFlightResponse):
            if response.booked_flight:
                st.write(f"Booked Flight ID: {response.booked_flight}")
            else:
                st.write("Flight booking unsuccessful.")

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.text})

# Display a warning about the simulated date
st.sidebar.warning("Note: This chatbot is simulating responses as if the current date is September 1, 2022.")