import streamlit as st
from datetime import datetime
import os
import yaml
from typing import List
from openai import OpenAI

# Importing your classes and functions
from Homework2 import (
    Flight,
    Agent,
    load_flights_dataset,
    eval_agent,
    SYSTEM_PROMPT,
    FindFlightsResponse,
    BookFlightResponse,
    TextResponse,
)

# Set up the OpenAI client
client = OpenAI(base_url=os.getenv("URL"), api_key=os.getenv("KEY"))

# Load the flights dataset
flights_dataset = load_flights_dataset()

# Title of the Streamlit app
st.title("Travel Agent LLM Interface")

# Step 1: Prompt for user inputs
st.sidebar.header("Flight Search")
origin = st.sidebar.text_input("Enter origin airport code", "BOS")
destination = st.sidebar.text_input("Enter destination airport code", "LAX")
departure_date = st.sidebar.date_input("Select departure date", datetime(2023, 9, 5))

# Initialize the agent globally to keep it accessible
if "agent" not in st.session_state:
    st.session_state.agent = Agent(
        flights=flights_dataset,
        client=client,
        conversation=[{"role": "system", "content": SYSTEM_PROMPT}],
    )

agent = st.session_state.agent  # Reference the global agent

# Step 2: Function to simulate agent response for search
if st.sidebar.button("Find Flights"):
    st.write(
        f"Searching for flights from {origin} to {destination} on {departure_date}"
    )

    # Agent finds flights
    find_flights_prompt = (
        f"I need to get from {origin} to {destination} on {departure_date}"
    )
    response = agent.say(find_flights_prompt)

    if isinstance(response, FindFlightsResponse):
        st.write("Available flights:")
        for flight_id in response.available_flights:
            flight = next(f for f in flights_dataset if f.id == flight_id)
            st.write(
                f"Flight ID: {flight.id}, Airline: {flight.airline}, Departure: {flight.departure_time}, Arrival: {flight.arrival_time}"
            )
    else:
        st.write("No flights found or an error occurred.")

# Step 3: Flight booking
book_flight_id = st.sidebar.number_input(
    "Enter flight ID to book", min_value=0, max_value=1000, value=0
)

if st.sidebar.button("Book Flight"):
    st.write(f"Attempting to book flight ID: {book_flight_id}")

    # Use the agent to book the flight
    book_prompt = f"book flight {book_flight_id}"
    response = agent.say(book_prompt)

    if isinstance(response, BookFlightResponse) and response.booked_flight:
        st.write(f"Flight {response.booked_flight} booked successfully!")
    else:
        st.write(f"Could not book flight {book_flight_id}")

# Step 4: Conversation Log
st.subheader("Conversation Log")
for message in agent.conversation:
    if message["role"] == "user":
        st.write(f"**User:** {message['content']}")
    elif message["role"] == "system":
        st.write(f"**System:** {message['content']}")
