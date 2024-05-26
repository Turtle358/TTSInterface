import torch
from transformers import AutoProcessor, BarkModel
import soundfile as sf
from pydub import AudioSegment
from autocorrect import Speller
import os


class TextToSpeech:
    def __init__(self, modelName):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelName = modelName
        self.processor = AutoProcessor.from_pretrained(modelName)
        self.model = BarkModel.from_pretrained(modelName, device_map=self.device)

    def textToSpeech(self, text):
        inputs = self.processor(text, return_tensors='pt')
        with torch.no_grad():
            speech = self.model.generate(**inputs)
        waveform = speech[0].cpu().numpy()
        return waveform

    def saveToFile(self, waveform, fileName, fileNum, fileTot, sampleRate=22050, ):
        sf.write(fileName, waveform, samplerate=sampleRate)
        print(f"{fileNum}/{fileTot}: Speech has been saved to {fileName}")

    def chunkText(self, text, chunkSize=512):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunkSize):
            chunks.append(' '.join(words[i:i + chunkSize]))
        return chunks

    def autocorrectText(self, text):
        spell = Speller(lang='en')
        correctedText = ' '.join(spell(word) for word in text.split())
        return correctedText

    def mergeSavedFiles(self):
        wavFiles = os.listdir('./rawOutputs')
        wavFiles = ["./rawOutputs/" + x for x in wavFiles]
        mergedAudio = AudioSegment.empty()

        for wavFile in wavFiles:
            audioSegment = AudioSegment.from_wav(wavFile)
            mergedAudio += audioSegment
        mergedAudio.export('./finalOutput.mp3', format='mp3')
        print('Saved to ./finalOutput.mp3')


if __name__ == "__main__":
    tts = TextToSpeech("ylacombe/bark-small")
    with open("input.txt", "r", encoding="utf-8") as file:
        text = file.read().lower()
    text = tts.autocorrectText(text)
    text = tts.chunkText(text, chunkSize=40)
    for i in range(len(text)):
        waveform = tts.textToSpeech(text[i])
        if not os.path.exists("./rawOutputs"):
            os.mkdir("./rawOutputs")
        tts.saveToFile(waveform, f"./rawOutputs/output_{i+1}.wav", i+1, len(text))
    tts.mergeSavedFiles()
