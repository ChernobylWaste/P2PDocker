import zmq
import os
import sys
import time
import hashlib
from threading import Thread

# Konfigurasi node dan port
NODES = ["node1", "node2", "node3", "node4"]
PORT = 65434

# Direktori untuk menyimpan file
FILE_DIR = "/files"

# Node saat ini
me = str(sys.argv[1])

# Fungsi untuk menghitung hash file
def calculate_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Fungsi untuk mengirim file
def send_file(filename, data, socket):
    socket.send_multipart([filename.encode(), data])

# Fungsi untuk menyimpan file
def save_file(filename, data):
    filepath = os.path.join(FILE_DIR, filename)
    if not os.path.exists(FILE_DIR):
        os.makedirs(FILE_DIR)

    if os.path.exists(filepath):
        # Jika file dengan nama sama ada, periksa hash
        existing_hash = calculate_file_hash(filepath)
        new_hash = hashlib.md5(data).hexdigest()
        if existing_hash == new_hash:
            print(f"File {filename} sudah ada, tidak di-overwrite.")
            return
    # Simpan file baru
    with open(filepath, "wb") as f:
        f.write(data)
    print(f"File {filename} disimpan.")

# Fungsi server untuk mengirim file
def server():
    context = zmq.Context()
    server_socket = context.socket(zmq.PUSH)
    server_socket.bind(f"tcp://*:{PORT}")
    print(f"Server di {me} berjalan...")
    
    known_files = {}

    while True:
        files_in_dir = os.listdir(FILE_DIR)
        for filename in files_in_dir:
            # Abaikan file sementara seperti .swp atau file tersembunyi
            if filename.endswith('.swp') or filename.startswith('.'):
                print(f"Skipping temporary file: {filename}")
                continue

            filepath = os.path.join(FILE_DIR, filename)
            file_hash = calculate_file_hash(filepath)
            if filename not in known_files or known_files[filename] != file_hash:
                # File baru atau file berubah
                with open(filepath, "rb") as f:
                    data = f.read()
                for node in NODES:
                    if node != me:
                        send_file(filename, data, server_socket)
                known_files[filename] = file_hash
                print(f"File {filename} dikirim ke node lainnya.")
        time.sleep(5)

# Fungsi klien untuk menerima file
def client():
    context = zmq.Context()
    client_socket = context.socket(zmq.PULL)
    for node in NODES:
        if node != me:
            client_socket.connect(f"tcp://{node}:{PORT}")
    print(f"Klien di {me} berjalan dan menunggu file...")

    while True:
        message = client_socket.recv_multipart()
        filename, data = message
        save_file(filename.decode(), data)

# Jalankan server dan klien secara paralel
if __name__ == "__main__":
    Thread(target=server).start()
    Thread(target=client).start()
