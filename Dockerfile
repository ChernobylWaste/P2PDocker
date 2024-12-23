FROM python:latest

RUN pip install pyzmq
RUN apt update
RUN apt install nano

# Copy file utama ke dalam image Docker
ADD main.py . 

# Tentukan direktori kerja
WORKDIR .

# Set volume untuk menyimpan file yang akan disinkronisasi
VOLUME ["/files"]
