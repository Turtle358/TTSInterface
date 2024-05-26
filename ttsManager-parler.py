import torch
from transformers import AutoTokenizer, set_seed
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
from pydub import AudioSegment
from autocorrect import Speller
import os


class TextToSpeech:
    def __init__(self, modelName):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(modelName).to(self.device)
        self.tokeniser = AutoTokenizer.from_pretrained(modelName)

    def textToSpeech(self, text, description):
        inputIDs = self.tokeniser(description, return_tensors="pt").input_ids.to(self.device)
        promptInputIDs = self.tokeniser(text, return_tensors="pt").input_ids.to(self.device)
        set_seed(42)
        generation = self.model.generate(input_ids=inputIDs, prompt_input_ids=promptInputIDs)
        audioArr = generation.cpu().numpy().squeeze()
        return audioArr

    def saveToFile(self, waveform, fileName, fileNum, fileTot):
        sf.write(fileName, waveform, samplerate=self.model.config.sampling_rate)
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
    tts = TextToSpeech("parler-tts/parler-tts-mini-jenny-30H")
    with open("input.txt", "r", encoding="utf-8") as file:
        text = file.read().lower()
    text = tts.autocorrectText(text)
    text = tts.chunkText(text, chunkSize=40)
    description = "Jenny speaks at an average pace with an animated delivery in a very confined sounding environment with clear audio quality."
    for i in range(len(text)):
        waveform = tts.textToSpeech(text[i], description)
        if not os.path.exists("./rawOutputs"):
            os.mkdir("./rawOutputs")
        tts.saveToFile(waveform, f"./rawOutputs/output_{i+1}.wav", i+1, len(text))
    tts.mergeSavedFiles()
