sentence = "What is a bag of words and what does it do for me when processing words?"
clean_text = sentence.lower().split(" ")
bow = {word: clean_text.count(word) for word in clean_text}
print(bow)