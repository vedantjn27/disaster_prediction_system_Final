import pyttsx3
class MyTTS:
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.setup_tts_engine()

    def setup_tts_engine(self):
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.8)

    def speak(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

# Usage
my_tts = MyTTS()
my_tts.speak("Hello, this is a test.")
