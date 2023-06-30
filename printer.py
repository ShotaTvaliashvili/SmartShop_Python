import qrcode
import cups

# Generate a QR code image
data = "Hello, World!"  # The data you want to encode in the QR code
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
qr.add_data(data)
qr.make(fit=True)
qr_img = qr.make_image(fill_color="black", back_color="white")

# Save the QR code image to a file
qr_img.save("qrcode.png")

# Print the QR code using the terminal printer
conn = cups.Connection()
print(conn, 'commm')
printers = conn.getPrinters()
print(printers)
printer_name = "gxmc_micro_printer"  # Replace with the actual name of your printer

# Check if the specified printer exists
if printer_name not in printers:
    print(f"Printer '{printer_name}' not found.")
    exit()

# Set the printer options
print_options = {
    "copies": '1',
    "media": "A4",
    "fit-to-page": 'true',
    "media-source": "auto",
    "printer-commands": "",
}

# Print the QR code image
print_job_title = "QR Code Print Job"
print_data = open("qrcode.png", "rb").read()
job_id = conn.printFile(printer_name, "qrcode.png", print_job_title, print_options)

print(f"Print job sent to printer '{printer_name}' with job ID: {job_id}")
