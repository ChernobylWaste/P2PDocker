FROM python:latest

RUN apt update && apt install -y nano
RUN apt update && apt install -y net-tools
RUN apt update && apt install -y iputils-ping

# Copy file utama ke dalam image Docker
ADD main.py . 

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fedavg.py .
COPY lokalml.py .
COPY Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv .

# Set volume untuk menyimpan file yang akan disinkronisasi
VOLUME ["/files"]