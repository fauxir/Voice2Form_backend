from flask import Flask, request
import whisper
from flask import jsonify
import time


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

if __name__ == "__main__":
    app.run()