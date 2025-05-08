import requests

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

# Save the file
with open("tinyshakespeare.txt", "w", encoding="utf-8") as f:
    f.write(response.text)

print("Download complete. Saved as 'tinyshakespeare.txt'")
