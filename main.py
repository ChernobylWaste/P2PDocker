import zmq
import time
import os
import sys

# Tentukan apakah ini server atau client berdasarkan argumen yang diberikan
me = str(sys.argv[1])
is_server = me == "node1"  # node1 bertindak sebagai server, lainnya sebagai client

context = zmq.Context()

# Server menggunakan PUSH socket untuk mengirim data ke client
if is_server:
    s = context.socket(zmq.PUSH)
    s.bind("tcp://*:65434")  # Server mendengarkan di port 65434

    # Fungsi untuk mengirim file ke client
    def send_file(filename, client_socket):
        with open(filename, 'rb') as f:
            data = f.read()
            # Kirimkan nama file dan data file ke client
            message = {"filename": filename, "data": data}
            client_socket.send_pyobj(message)

    server_dir = "/files"  # Lokasi folder tempat file akan disimpan di server
    while True:
        files_in_server = os.listdir(server_dir)  # List file yang ada di server

        # Kirimkan file ke semua client
        for filename in files_in_server:
            # Abaikan file dengan ekstensi .swp atau file tersembunyi
            if filename.startswith('.') or filename.endswith('.swp'):
                continue

            if filename != "received_file.txt":  # Menghindari pengiriman file yang sedang diterima
                for client in ["tcp://node2:65434", "tcp://node3:65434", "tcp://node4:65434"]:
                    print(f"Sending {filename} to {client}")
                    send_file(f"{server_dir}/{filename}", s)
        time.sleep(5)  # Cek setiap 5 detik untuk file baru

# Client menggunakan PULL socket untuk menerima data dari server
else:
    s = context.socket(zmq.PULL)
    s.connect("tcp://node1:65434")  # Client terhubung ke server

    # Fungsi untuk menyimpan file yang diterima
    def save_file(filename, data):
        file_path = f"/{filename}"  # Menyimpan file dengan nama yang sesuai tanpa duplikasi '/files/'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Pastikan folder '/files/' ada
        with open(file_path, 'wb') as f:
            f.write(data)
        print(f"File received and saved to {file_path}")

    # Client menerima file dari server
    while True:
        message = s.recv_pyobj()  # Menerima objek yang berisi nama file dan data
        filename = message["filename"]  # Mendapatkan nama file yang dikirim
        data = message["data"]  # Mendapatkan data file

        # Menyimpan file dengan nama yang sesuai
        save_file(filename, data)  # Simpan file dengan nama yang sesuai dari server
