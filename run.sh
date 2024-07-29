#!/bin/bash

# # Open the first terminal and run run1.sh
# gnome-terminal -- bash -c "./run1.sh; exec bash"

# # Open the second terminal and run run2.sh
# gnome-terminal -- bash -c "./run2.sh; exec bash"



# Open the first terminal and run run1.sh
start bash -c "./run1.sh; exec bash"

sleep 5

# Open the second terminal and run run2.sh
start bash -c "./run2.sh; exec bash"