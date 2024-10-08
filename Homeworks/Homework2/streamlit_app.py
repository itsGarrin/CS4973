import logging
import os
from datetime import datetime, date

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from Homework2 import (
    load_flights_dataset,
    Agent,
    FindFlightsResponse,
    BookFlightResponse,
    CONVO
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and initialize data
flights_data = load_flights_dataset()
client = OpenAI(base_url=os.getenv("URL"), api_key=os.getenv("KEY"))

def main():
    flight_map = {flight.id: flight for flight in flights_data}

    st.set_page_config(page_title="Travel Agent LLM", layout="wide")
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>✈️ Travel Agent LLM</h1>", unsafe_allow_html=True)

    logging.info("Streamlit application started.")

    st.markdown(
        """
        <style>
        .sidebar .sidebar-content { background-color: #f0f2f6; padding: 20px; }
        .stButton>button { background-color: #4CAF50; color: white; }
        .stButton>button:hover { background-color: #45a049; }
        .available-flights { margin-top: 20px; }
        .message { border-radius: 10px; margin: 10px 0; padding: 10px; color: #333; }
        .user-message { background-color: #f1f0ff; }
        .agent-message { background-color: #ffefef; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Search for Flights")
        origin = st.text_input("Origin (Airport Code)", max_chars=3, placeholder="e.g., ATL")
        destination = st.text_input("Destination (Airport Code)", max_chars=3, placeholder="e.g., SEA")
        travel_date = st.date_input("Travel Date", min_value=date(2023, 1, 1), max_value=date(2023, 12, 31))
        find_btn = st.button("Find Flights", use_container_width=True)

        st.header("Book a Flight")
        flight_id = st.number_input("Flight ID to Book", value=1, min_value=1)
        book_btn = st.button("Book Flight", use_container_width=True)


    agent = Agent(conversation=CONVO.copy(), flights=flights_data, client=client)
    logging.info("Agent initialized.")

    col1, col2 = st.columns([2, 3])

    with col1:
        if find_btn:
            st.header("Available Flights")
            logging.info(f"Find Flights button clicked with Origin: {origin}, Destination: {destination}, Date: {travel_date}")
            if origin and destination:
                if travel_date.year != 2023:
                    st.error("Please select a date within the year 2023.")
                    logging.warning("Date selected is not within the year 2023.")
                else:
                    formatted_date = datetime.combine(travel_date, datetime.min.time())
                    query = f"Find me a flight from {origin} to {destination} on {formatted_date.strftime('%B %d, %Y')}."
                    with st.spinner('Searching for flights...'):
                        logging.debug(f"Sending agent query: {query}")
                        response = agent.say(query)
                    logging.debug(f"Agent response received: {response}")

                    if isinstance(response, FindFlightsResponse):
                        if response.available_flights:
                            for flight_id in response.available_flights:
                                flight = flight_map.get(flight_id)
                                st.write(f"✈️ **Flight ID**: {flight.id}, **Airline**: {flight.airline}, **Flight Number**: {flight.flight_number}, **Origin**: {flight.origin}, **Destination**: {flight.destination}, **Departure**: {flight.departure_time}, **Arrival**: {flight.arrival_time}, **Seats**: {flight.available_seats}")
                        else:
                            st.warning("No flights found.")
                            logging.info("No flights found for the given search criteria.")
                    else:
                        st.warning(response.text or "An unexpected error occurred. Please try again.")
                        logging.error(f"Unexpected response: {response.text}")
            else:
                st.error("Please enter both an origin and a destination.")
                logging.warning("Either origin or destination input is missing.")

    with col2:
        if book_btn:
            st.header("Booking Confirmation")
            logging.info(f"Book Flight button clicked with Flight ID: {flight_id}")
            if flight_id > 0:
                with st.spinner('Booking your flight...'):
                    response = agent.say(f"Book the flight with an ID of {flight_id}.")
                logging.debug(f"Agent response for booking: {response}")

                if isinstance(response, BookFlightResponse):
                    booked_flight = response.booked_flight
                    if booked_flight:
                        st.success(f"Flight with ID {booked_flight} booked successfully!")
                        logging.info(f"Flight with ID {booked_flight} booked successfully.")
                    else:
                        st.error("Failed to book the flight. No available seats or invalid flight ID.")
                        logging.error("Booking failed - no available seats or invalid flight ID.")
                else:
                    st.error(response.text or "An unexpected error occurred during booking. Please try again.")
                    logging.error(f"Unexpected response during booking: {response.text}")
            else:
                st.error("Please enter a valid flight ID greater than zero.")
                logging.warning("Invalid flight ID entered.")

    st.subheader("Conversation History")
    conversation_display = ""
    for i, turn in enumerate(CONVO.copy()):
        role = "User" if turn["role"] == "user" else "Agent"
        class_name = "user-message" if turn["role"] == "user" else "agent-message"
        conversation_display += f"<div class='{class_name} message'><strong>{role}:</strong> {turn['content']}</div>"
    st.markdown(conversation_display, unsafe_allow_html=True)

if __name__ == "__main__":
    main()