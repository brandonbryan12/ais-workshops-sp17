from gtts import gTTS
from transcribe import transcribe_file
import os
import sys
import subprocess
import neural_net as nn

def getTextFromSpeech(filePath):
    text = subprocess.check_output([sys.executable, "transcribe.py", filePath])
    return text[text.index(':') + 1 :]

def runBot(request):
    return request

def speakResponse(response):
    #tts = gTTS(text=response, lang='en')
    tts = gTTS(text=response, lang='en')
    tts.save("resources/bot_output.wav")
    os.system("start resources/bot_output.wav")

if __name__ == '__main__':
    nn.train()
    filename = sys.argv[1]
    request = getTextFromSpeech(filename)
    responseText = nn.response(request)
    response = runBot(request)
    speakResponse(response)
