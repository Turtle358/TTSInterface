import torch
from transformers import AutoProcessor, BarkModel
import soundfile as sf
import numpy as np


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

    def saveToFile(self, waveform, file_name, sample_rate=22050):
        sf.write(file_name, waveform, samplerate=sample_rate)
        print(f"Speech has been saved to {file_name}")


if __name__ == "__main__":
    tts = TextToSpeech("yefengzi/bark-small-fork")
    text = "Hello Artem, this is from text to speech using yefengzi/bark-small-fork."
    waveform = tts.textToSpeech(text)
    tts.saveToFile(waveform, "output.wav")
