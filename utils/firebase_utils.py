import os

def initialize_firebase():
    firebase_cred_path = "D:/coba/facerecognition-c8264-firebase-adminsdk-nodyk-90850d2e73.json"
    
    if not os.path.exists(firebase_cred_path):
        print("File kredensial tidak ditemukan! Pastikan pathnya benar.")
        return

    # Path ke file kredensial Firebase Anda
    cred = credentials.Certificate(firebase_cred_path)
    
    try:
        # Inisialisasi Firebase Admin SDK
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'facerecognition-c8264.appspot.com',  # Ganti dengan ID bucket Firebase Anda
            'databaseURL': 'https://facerecognition-c8264-default-rtdb.firebaseio.com/'  # Ganti dengan URL database Firebase Anda
        })
        print("Firebase has been initialized.")
    except Exception as e:
        print(f"Gagal menginisialisasi Firebase: {e}")

# Panggil fungsi untuk menginisialisasi Firebase
initialize_firebase()
