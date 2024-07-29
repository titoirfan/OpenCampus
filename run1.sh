#!/bin/bash

# Run the Python script in the background
env/Scripts/activate
python src/RL_server.py &
