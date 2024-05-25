from transformers import pipeline
import numpy as np
import soundfile as sf
import torch
import os


class ttsManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline('text-to-speech', model="suno/bark-small", device=self.device)
        print("Model loaded successfully")

    def generateAudio(self, text):
        output = self.pipe(text)
        print(f'Phrase "{text}" successfully generated')
        return output

    def saveAudio(self, output, outputFile='./output.mp3'):
        # Assuming output is a dictionary containing the audio samples and sample rate
        audio_samples = output["audio"]
        sample_rate = 44000

        # Convert the audio samples to a numpy array
        audio_data = np.array(audio_samples)

        # Save the audio data to a WAV file
        sf.write(outputFile, audio_data, sample_rate, format='mp3')
        print(f'Audio successfully saved to {outputFile}')


if __name__ == '__main__':
    tts = ttsManager()
    output = tts.generateAudio('Hello world')
    tts.saveAudio(output)
