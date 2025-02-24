from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

@app.route("/voice", methods=["POST"])
def voice():
    """Handles incoming calls and responds with the company name"""
    response = VoiceResponse()
    response.say("Hello! Welcome to XYZ Company. How can I assist you today?", voice="alice")
    return str(response)

if __name__ == "__main__":
    app.run(port=5000)
