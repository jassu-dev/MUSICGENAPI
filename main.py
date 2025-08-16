from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from transformers import pipeline
from pydub import AudioSegment
import numpy as np
import os
import tempfile

app = Flask(__name__)
CORS(app)

@app.route('/')
def root():
    return jsonify({"message": "Music Generation API. Use POST /generate_music with prompt and style."})

@app.route('/generate_music', methods=['POST'])
def generate_music():
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        output_file = temp_file.name

    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json()
        prompt = data.get('prompt')
        style = data.get('style')
        if not prompt or not style:
            return jsonify({"error": "Missing prompt or style"}), 400
        text_input = f"{style} {prompt}"
        synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")
        music = synthesiser(text_input, forward_params={"do_sample": True})
        audio_data = (music["audio"] * 32767).astype(np.int16)
        audio_data = np.squeeze(audio_data)
        channels = 1 if len(audio_data.shape) == 1 else 2
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=music["sampling_rate"],
            sample_width=2,
            channels=channels
        )
        audio_segment.export(output_file, format="mp3", bitrate="192k")

        # Send file as response
        return send_file(
            output_file,
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name="generated_music.mp3"
        )

    except Exception as e:
        return jsonify({"error": f"Error generating music: {str(e)}"}), 500
    finally:
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
        except PermissionError:
            print(f"Could not delete {output_file}: File in use. Will be cleaned up later.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)