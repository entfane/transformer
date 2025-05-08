import pickle

def extract_char_vocab(text):
    return set(text)

def extract_vocab_from_readings(files):
    vocab = set()
    for file in files:
        with open(file, "r") as reader:
            lines = reader.readlines()
            for line in lines:
                vocab.update(extract_char_vocab(line))

    return list(vocab)

vocab = extract_vocab_from_readings(["tinyshakespeare.txt"])
char_dict = {index: char for index, char in enumerate(vocab)}
print(char_dict)
with open('idx_to_token.pkl', 'wb') as f:
    pickle.dump(char_dict, f)

char_idx = dict((v,k) for k,v in char_dict.items())
print(char_idx)

with open('token_to_idx.pkl', 'wb') as f:
    pickle.dump(char_idx, f)


