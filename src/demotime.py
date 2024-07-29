import time
import csv
from datetime import datetime
import tkinter as tk

def get_next_time(current_time, time_list):
    for t in time_list:
        if t > current_time:
            return t
    return None

def update_display():
    with open('src/DemoTime.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        time_list = [row[0] for row in reader]
    current_time = time.strftime("%H:%M:%S", time.localtime())
    # current_time = "10:00:00"
    next_time = get_next_time(current_time, time_list)
    
    # Extract hours and minutes from current time
    current_hour_minute = datetime.now().strftime("%H:%M")
    
    if next_time:
        # Remove seconds from next_time
        next_time_hour_minute = next_time.rsplit(':', 1)[0]
        display_text = f"Current time: {current_hour_minute}\n \nNEXT DEMO\n{next_time_hour_minute}"
    else:
        display_text = f"Current time: {current_hour_minute}\nFinished"
    
    label.config(text=display_text)
    root.after(10000, update_display)  # Update every 60 seconds

# Read times from CSV
with open('src/DemoTime.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    time_list = [row[0] for row in reader]

# Set up the Tkinter window
root = tk.Tk()
root.title("Demo Time Display")
label = tk.Label(root, text="", font=("Helvetica", 150), justify="center")
label.pack(expand=True)

# Initial display update
update_display()

# Start the Tkinter main loop
root.mainloop()