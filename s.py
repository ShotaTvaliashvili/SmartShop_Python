import serial

# Define the serial port and baud rate
ser = serial.Serial('/dev/cu.usbserial-110', 57600) # Update the port and baud rate as per your setup

# Read and print the weight data
while True:
    if ser.is_open:
        # Read a line of data from the serial port
        line = ser.readline().decode().strip()
        print(line)
        # Check if the line contains weight data
        if line.startswith("Weight:"):
            # Extract the weight value from the line
            weight = line.split(":")[1].strip()
            
            # Print the weight value
            print("Weight:", weight)

