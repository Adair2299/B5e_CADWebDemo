"""Lossless printable-ASCII character tokenizer, shared by train and inference."""
import csv
import json
from pathlib import Path
from cad_schema import canonical, parse_command

SPECIAL_TOKENS = ['<pad>', '<bos>', '<sep>', '<eos>', '<unk>']
TOKENIZER_VERSION = 'ascii-char-v1'
def tokenize(text):
    if any(not 32 <= ord(char) <= 126 for char in text):
        raise ValueError('Use printable ASCII English input.')
    return list(text)
def build_vocabulary(pairs=None):
    return {token: i for i, token in enumerate(SPECIAL_TOKENS + [chr(i) for i in range(32, 127)])}
def encode(text, vocabulary):
    return [vocabulary[char] for char in tokenize(text)]
def detokenize(tokens):
    return ''.join(tokens)
def question_key(question):
    return ' '.join(question.lower().split())
def load_pairs(paths):
    result = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ['question', 'answer']:
                raise ValueError('Expected question,answer columns: ' + str(path))
            for row in reader:
                question = row['question'].strip()
                if not question:
                    raise ValueError('Empty question.')
                tokenize(question)
                answer = canonical(parse_command(row['answer']))
                key = question_key(question)
                if key in result and result[key][1] != answer:
                    raise ValueError('Conflicting labels: ' + question)
                result[key] = (question, answer)
    if not result:
        raise ValueError('No CAD data found. Generate Dataset 1 or provide Dataset 2 explicitly; training was not started.')
    return list(result.values())
