import os

CORPUS_PATH = "../Corpora/Buckeye corpus/words_tagged/"
files = [CORPUS_PATH+file for file in os.listdir(CORPUS_PATH) if ".tagged" in file]
files.sort()

phonemes_file = open(CORPUS_PATH+"phonemes.txt", 'w')
text_file = open(CORPUS_PATH+"words.txt", 'w')

words_buffer = []
phonemes_buffer = []

for file in files:
    for line in open(file, 'r').readlines():
        bits = line.split(';')
        if len(bits) < 3:
            continue
        word = bits[0].split(' ')[-1]
        phonemes = bits[1].strip()
        if word[-1] in [">", "}"] and words_buffer != []:
            phonemes_file.write(" ;eword ".join(phonemes_buffer)+" ;eword\n")
            text_file.write(" ".join(words_buffer) + "\n")
            words_buffer = []
            phonemes_buffer = []
        elif not word[-1] in [">", "}"]:
            words_buffer.append(word)
            phonemes_buffer.append(phonemes)
    if words_buffer != []:
        phonemes_file.write(" ;eword ".join(phonemes_buffer)+" ;eword\n")
        text_file.write(" ".join(words_buffer) + "\n")
        words_buffer = []
        phonemes_buffer = []

phonemes_file.close()
text_file.close()
