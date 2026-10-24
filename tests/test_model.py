"""Forward-only tests: no training or checkpoint creation."""
import unittest
import torch
from model import TinyChatbot
from train import QADataset, split_pairs, check_disjoint
from data_utils import build_vocabulary, encode
from generate_dataset_1 import generate_pairs
from cad_schema import canonical

class ModelTests(unittest.TestCase):
    def test_causal_mask(self):
        torch.manual_seed(2)
        model = TinyChatbot(100, max_length=32, dropout=0).eval()
        a = torch.tensor([[1, 2, 3, 4]])
        b = torch.tensor([[1, 2, 9, 8]])
        with torch.no_grad():
            out_a, out_b = model(a), model(b)
        self.assertEqual(tuple(out_a.shape), (1, 4, 100))
        torch.testing.assert_close(out_a[:, :2], out_b[:, :2])

    def test_context_bounds(self):
        model = TinyChatbot(100, max_length=4)
        with self.assertRaises(ValueError):
            model(torch.ones((1, 5), dtype=torch.long))

    def test_answer_only_targets_and_padding(self):
        v = build_vocabulary()
        a = canonical({'command': 'UNDO'})
        dataset = QADataset([('undo', a), ('please undo', a)], v, 100)
        x, y = dataset[0]
        self.assertEqual(y[:5], [-100] * 5)
        self.assertEqual(y[5], encode(a, v)[0])
        inputs, targets = dataset.collate([dataset[0], dataset[1]])
        self.assertEqual(int(targets[0, -1]), -100)
        self.assertEqual(int(inputs[0, -1]), v['<pad>'])
        with self.assertRaises(ValueError):
            QADataset([('undo', a)], v, 5)

    def test_grouped_partitions(self):
        parts = split_pairs(generate_pairs(200), 42)
        check_disjoint(parts)
        for i in range(3):
            for j in range(i):
                self.assertFalse({a for _, a in parts[i]} & {a for _, a in parts[j]})
        with self.assertRaises(ValueError):
            check_disjoint(([('same', 'a')], [('Same', 'a')], []))

if __name__ == '__main__':
    unittest.main()
