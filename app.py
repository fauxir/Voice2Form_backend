from flask import Flask, request, jsonify
import whisper
import time
import sqlite3
import random
import string
import os
import json
import shutil 



app = Flask(__name__)

@app.route("/", methods=["POST"])
def transcribe():
    if "recording" in request.files:
        recording_file = request.files["recording"]
    # Save or process the recording file as needed
        recording_file.save("./recording/recording.m4a")
    start_time = time.time()  # Get the current time at the start of the function
    model = whisper.load_model("tiny")
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("./recording/recording.m4a")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False, language='en')
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)
    
    end_time = time.time()  # Get the current time after the desired operations

    execution_time = end_time - start_time  # Calculate the execution time in seconds
    data = {
        "result": result.text,
    }
    print(f"Execution time: {execution_time} seconds")  # Print the execution time to the console
    return jsonify(data)


db_path = "database/main.db"
db_backup_path = "database/main_backup.db"  # Specify the backup file name and path

# Function to create the database and table if they don't exist
def create_db_table():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS YourTable (ID INTEGER PRIMARY KEY AUTOINCREMENT, Reference TEXT, Object TEXT)"
    )
    conn.commit()
    conn.close()

@app.route("/store_object", methods=["POST"])
def store_object():
    try:
        # Backup the database before any write operations
        if os.path.exists(db_path):
            shutil.copy2(db_path, db_backup_path)  # Create a backup copy

        # Check if the database exists, if not, create it
        if not os.path.exists(db_path):
            create_db_table()

        data = request.json

        # Generate a random 10-digit alphanumeric reference
        reference = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

        # Store the object and reference in the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO YourTable (Reference, Object) VALUES (?, ?)",
                       (reference, json.dumps(data)))
        conn.commit()
        conn.close()

        return jsonify({"message": "Object stored successfully", "reference": reference}), 201
    except Exception as e:
        # If an error occurs during write operations, restore the backup
        if os.path.exists(db_backup_path):
            shutil.copy2(db_backup_path, db_path)  # Restore from backup
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)