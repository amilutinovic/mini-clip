import re
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

def normalize_text(text):
    text = text.lower().strip()
    return text
    
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)
    
def encode_text(text, vocab):
	tokens = tokenize(text)

	tokens = [
	SOS_TOKEN,
	*tokens,
	EOS_TOKEN
	]

	return [
	vocab.token_to_id.get(token, vocab.token_to_id[UNK_TOKEN])
	for token in tokens
	]
def pad_or_truncate(ids, max_length, pad_id):
    if len(ids) > max_length:
        ids = ids[:max_length]

        # Ensure that the sequence ends with EOS.
        ids[-1] = vocab.token_to_id[EOS_TOKEN]

    else:
        ids = ids + [pad_id] * (max_length - len(ids))

    return ids
