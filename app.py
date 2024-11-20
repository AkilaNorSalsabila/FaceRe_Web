from flask import Flask, render_template, request, redirect, jsonify, session
import os
import cv2
import numpy as np
import base64
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from firebase_admin import credentials, initialize_app, db

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Ganti dengan secret key yang aman
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # Maksimum 64 MB

# Inisialisasi Firebase
cred = credentials.Certificate("D:/coba/facerecognition-c8264-firebase-adminsdk-nodyk-90850d2e73.json")
initialize_app(cred, {
    'databaseURL': 'https://facerecognition-c8264-default-rtdb.firebaseio.com/',
})

# Path dataset
dataset_path = 'DataSet'

# Halaman Login Admin
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Cek login di Firebase
        ref = db.reference('akun')
        accounts = ref.get()

        for user in accounts.values():
            if user['username'] == username and check_password_hash(user['password'], password):
                session['user'] = username
                return redirect('/dashboard')

        return render_template('index.html', message="Username atau Password salah!")

    return render_template('index.html')

# Halaman Dashboard Admin
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')
    return render_template('dashboard.html')

# Route untuk menambahkan dataset
@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    if request.method == 'POST':
        try:
            name = request.form['name']
            nim = request.form['nim']
            student_id = f'ID-{nim}_{name}'

            if not all([name, nim]):
                return jsonify({'status': 'error', 'message': 'Nama dan NIM harus diisi!'}), 400

            # Buat folder untuk menyimpan gambar
            folder_path = os.path.join(dataset_path, student_id)
            os.makedirs(folder_path, exist_ok=True)

            # Hitung jumlah gambar di folder
            image_count = len(os.listdir(folder_path))
            if image_count >= 10:
                return jsonify({'status': 'error', 'message': 'Jumlah maksimum gambar telah tercapai.'}), 400

            # Decode gambar Base64
            image_data = request.form['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            img_data = base64.b64decode(image_data)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Gambar tidak valid atau kosong.")

            # Simpan gambar di lokal
            file_name = f'{student_id}_{image_count + 1}.jpg'
            file_path = os.path.join(folder_path, file_name)
            cv2.imwrite(file_path, img)

            # Simpan metadata mahasiswa ke Firebase
            student_ref = db.reference(f'students/{student_id}')
            student_data = student_ref.get()

            if not student_data:
                student_ref.set({
                    'id': student_id,
                    'name': name,
                    'nim': nim,
                    'timestamp': datetime.now().isoformat()
                })

            return jsonify({'status': 'success', 'message': 'Gambar berhasil disimpan.', 'student_id': student_id})

        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Terjadi kesalahan: {str(e)}'}), 500

    return render_template('dataset.html')

# Route untuk mengambil data mahasiswa
@app.route('/students', methods=['GET'])
def get_students():
    if 'user' not in session:
        return redirect('/')

    try:
        # Ambil data mahasiswa dari Firebase
        students_ref = db.reference('students')
        students_data = students_ref.get()

        students = []
        if students_data:
            for student_id, student_info in students_data.items():
                # Hitung jumlah gambar di folder lokal
                folder_path = os.path.join(dataset_path, student_id)
                images_count = len(os.listdir(folder_path)) if os.path.exists(folder_path) else 0

                students.append({
                    'id': student_info.get('id', 'Unknown'),
                    'name': student_info.get('name', 'Unknown'),
                    'nim': student_info.get('nim', 'Unknown'),
                    'images': images_count
                })

        return jsonify({'status': 'success', 'data': students})
    except Exception as e:
        print(f"Error saat mengambil data mahasiswa dari Firebase: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Halaman Training Model
@app.route('/train', methods=['GET', 'POST'])
def train():
    if 'user' not in session:
        return redirect('/')

    if request.method == 'POST':
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
            import json
            import os

            # Konfigurasi dataset
            dataset_path = "DataSet"
            img_size = (224, 224)
            batch_size = 32

            # Augmentasi Data
            train_gen = ImageDataGenerator(
                rescale=1.0 / 255,
                validation_split=0.2,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                brightness_range=[0.8, 1.2],
                horizontal_flip=True,
                fill_mode="nearest"
            )

            # Data training dan validasi
            train_data = train_gen.flow_from_directory(
                dataset_path,
                target_size=img_size,
                color_mode='rgb',
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )

            valid_data = train_gen.flow_from_directory(
                dataset_path,
                target_size=img_size,
                color_mode='rgb',
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )

            # Jumlah kelas
            num_classes = train_data.num_classes

            # Model MobileNetV2
            mobilenet = tf.keras.applications.MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )

            for layer in mobilenet.layers[:-10]:
                layer.trainable = False

            # Tambahkan lapisan klasifikasi
            model = models.Sequential([
                mobilenet,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation='softmax')
            ])

            # Kompilasi model
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            checkpoint_path = "models/best_model_mobilenet.keras"
            checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)

            # Pelatihan model
            model.fit(
                train_data,
                validation_data=valid_data,
                epochs=20,
                callbacks=[early_stopping, checkpoint]
            )

            # Simpan model akhir
            final_model_path = "models/final_trained_model.keras"
            model.save(final_model_path)

            # Simpan label map
            label_map = train_data.class_indices
            label_map = {v: k for k, v in label_map.items()}
            with open('label_map.json', 'w') as f:
                json.dump(label_map, f)

            return render_template('train.html', message="Model berhasil dilatih dan disimpan!")

        except Exception as e:
            return render_template('train.html', message=f"Terjadi kesalahan saat pelatihan: {str(e)}")

    return render_template('train.html', message=None)


# Route logout
@app.route('/logout')
def logout():
    # Hapus sesi pengguna
    session.pop('user', None)
    return redirect('/')

# Jalankan aplikasi Flask
if __name__ == "__main__":
    app.run(debug=True, port=5000)
