"""Greedy autoregressive Transformer decoding; no rule-based fallback."""
from pathlib import Path
import torch
from model import TinyChatbot
from data_utils import TOKENIZER_VERSION, build_vocabulary, encode
from cad_schema import VERSION

class Predictor:
    def __init__(self, checkpoint='cad_model.pt', device='cpu'):
        if not Path(checkpoint).is_file():
            raise ValueError('CAD model is not trained. Train cad_model.pt or use cad_cli.py --executor-test.')
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved.get('schema_version') != VERSION or saved.get('tokenizer_version') != TOKENIZER_VERSION:
            raise ValueError('Incompatible CAD checkpoint. Original model.pt is a chatbot, not a CAD model.')
        self.vocabulary = saved['vocabulary']
        if self.vocabulary != build_vocabulary():
            raise ValueError('Checkpoint vocabulary is incompatible.')
        self.inverse = {i: t for t, i in self.vocabulary.items()}
        self.device = device
        self.model = TinyChatbot(**saved['model_config']).to(device)
        self.model.load_state_dict(saved['model_state'])
        self.model.eval()

    @torch.no_grad()
    def predict(self, question):
        v = self.vocabulary
        tokens = [v['<bos>']] + encode(question, v) + [v['<sep>']]
        limit = self.model.config['max_length']
        if len(tokens) >= limit:
            raise ValueError('Question is too long for checkpoint context.')
        output = []
        while len(tokens) < limit:
            next_id = int(self.model(torch.tensor([tokens], device=self.device))[0, -1].argmax())
            if next_id == v['<eos>']:
                return ''.join(output)
            token = self.inverse[next_id]
            if token.startswith('<') and len(token) > 1:
                raise ValueError('Model generated an invalid special token.')
            output.append(token)
            tokens.append(next_id)
        raise ValueError('Generation reached context limit before EOS.')
