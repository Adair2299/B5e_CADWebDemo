"""Behavior tests use only in-memory pairs and geometry."""
import json
import math
import unittest
import io
from unittest.mock import patch
from copy import deepcopy
from cad_schema import SCHEMAS, CommandError, canonical, parse_command, validate_command
from executor import Executor
from generate_dataset_1 import generate_pairs
from data_utils import build_vocabulary, encode, detokenize
from data_utils import load_pairs
from evaluate import evaluate_cases

def command(name, **fields):
    return dict(command=name, **fields)

class SchemaTests(unittest.TestCase):
    def test_csv_loader_missing_invalid_conflicts(self):
        with patch('pathlib.Path.exists', return_value=False):
            with self.assertRaisesRegex(ValueError, 'No CAD data'):
                load_pairs(['dataset_1.csv', 'dataset_2.csv'])
        import csv
        def csv_text(rows):
            handle = io.StringIO()
            writer = csv.writer(handle)
            writer.writerow(['question', 'answer'])
            writer.writerows(rows)
            return handle.getvalue()
        good = csv_text([('undo', canonical(command('UNDO')))] * 2)
        with patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.open', side_effect=lambda **kwargs: io.StringIO(good)):
            self.assertEqual(len(load_pairs(['one.csv', 'two.csv'])), 1)
        bad = csv_text([('undo', canonical(command('UNDO'))), ('Undo', canonical(command('CLEAR_CANVAS')))])
        with patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.open', side_effect=lambda **kwargs: io.StringIO(bad)):
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                load_pairs(['conflict.csv'])

    def test_twenty_commands(self):
        self.assertEqual(len(SCHEMAS), 20)

    def test_generation_coverage_and_reproducibility(self):
        for split in ('train', 'validation', 'test'):
            pairs = generate_pairs(400, 11, split)
            self.assertEqual(pairs, generate_pairs(400, 11, split))
            self.assertEqual(len({q.lower() for q, _ in pairs}), 400)
            self.assertEqual({parse_command(a)['command'] for _, a in pairs}, set(SCHEMAS))
            for q, a in pairs:
                self.assertEqual(canonical(parse_command(a)), a)

    def test_split_templates_disjoint(self):
        parts = [{q.lower() for q, _ in generate_pairs(200, 1, split)} for split in ('train', 'validation', 'test')]
        for i in range(3):
            for j in range(i):
                self.assertFalse(parts[i] & parts[j])

    def test_lossless_tokenizer(self):
        v = build_vocabulary()
        inverse = {i: t for t, i in v.items()}
        for text in ['Move line 29 by -12.375, +0.25.', '{"command":"SET_LENGTH","id":2,"value":-0.125}', '1e-06']:
            self.assertEqual(detokenize([inverse[i] for i in encode(text, v)]), text)
        with self.assertRaises(ValueError):
            encode('\n', v)

    def test_bad_schema(self):
        bad = [command('SET_LENGTH', id=True, value=2), command('SET_WIDTH', id=0, value=1),
               command('CREATE_CIRCLE', x=0, y=0, radius=-1), command('UNDO', extra=1),
               command('MOVE_OBJECT', id=1, dx=float('inf'), dy=0),
               command('SET_COLOR', id=1, color='purple'), command('GET_POSITION', id=1.5),
               command('COUNT_IF', type='all', property='color', operator='gt', value='red'),
               command('COUNT_IF', type='line', property='length', operator='eq', value='six')]
        for c in bad:
            with self.subTest(c=c), self.assertRaises(CommandError):
                validate_command(c)
        for text in ['[]', '{"command":"UNDO","command":"UNDO"}', '{"command":"SET_WIDTH","id":1,"value":NaN}', 'not json']:
            with self.assertRaises(CommandError):
                parse_command(text)

class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.e = Executor()
        self.e.execute(command('CREATE_LINE', x1=0, y1=0, x2=3, y2=4))

    def test_creation_all_types(self):
        self.e.execute(command('CREATE_RECTANGLE', x=1, y=2, width=3, height=4))
        self.e.execute(command('CREATE_CIRCLE', x=-2, y=.5, radius=4))
        self.e.execute(command('CREATE_POINT', x=0, y=0))
        self.assertEqual(self.e.execute(command('COUNT_OBJECTS', type='all')), 4)
        for t in ('line', 'rectangle', 'circle', 'point'):
            self.assertEqual(self.e.execute(command('COUNT_OBJECTS', type=t)), 1)
        self.assertEqual(self.e.objects[2]['geometry_width'], 3)

    def test_length_move_rotate(self):
        self.e.execute(command('SET_LENGTH', id=1, value=10))
        self.assertEqual(self.e.execute(command('GET_LENGTH', id=1)), 10)
        self.e.execute(command('MOVE_OBJECT', id=1, dx=-2.5, dy=1))
        self.assertEqual(self.e.execute(command('GET_POSITION', id=1)), {'x': -2.5, 'y': 1})
        self.e.execute(command('ROTATE_OBJECT', id=1, angle=90))
        self.assertAlmostEqual(self.e.objects[1]['x2'], -10.5)
        self.assertAlmostEqual(self.e.objects[1]['y2'], 7)
        self.assertAlmostEqual(self.e.execute(command('GET_LENGTH', id=1)), 10)

    def test_rectangle_rotation_and_width(self):
        self.e.execute(command('CREATE_RECTANGLE', x=0, y=0, width=3, height=4))
        self.e.execute(command('ROTATE_OBJECT', id=2, angle=-90))
        self.e.execute(command('SET_WIDTH', id=2, value=.25))
        self.assertEqual(self.e.objects[2]['angle'], 270)
        self.assertEqual(self.e.objects[2]['geometry_width'], 3)
        self.assertEqual(self.e.objects[2]['width'], .25)

    def test_colors_conditions_and_queries(self):
        self.e.execute(command('SET_COLOR', id=1, color='red'))
        self.e.execute(command('SET_WIDTH', id=1, value=.5))
        for op, value, expected in [('eq', 5, 1), ('ne', 5, 0), ('gt', 4, 1), ('ge', 5, 1), ('lt', 5, 0), ('le', 5, 1)]:
            self.assertEqual(self.e.execute(command('COUNT_IF', type='line', property='length', operator=op, value=value)), expected)
        self.assertEqual(self.e.execute(command('FIND_OBJECTS', type='all', property='color', operator='eq', value='red')), [1])
        self.assertEqual(self.e.execute(command('COUNT_IF', type='line', property='radius', operator='eq', value=5)), 0)
        obj = self.e.execute(command('GET_PROPERTIES', id=1))
        obj['color'] = 'blue'
        self.assertEqual(self.e.objects[1]['color'], 'red')

    def test_selection_duplicate_delete_clear_undo(self):
        self.e.execute(command('SELECT_OBJECT', id=1))
        self.e.execute(command('DUPLICATE_OBJECT', id=1, dx=2, dy=-1))
        self.assertEqual(self.e.objects[2]['x1'], 2)
        self.e.execute(command('DELETE_OBJECT', id=1))
        self.assertIsNone(self.e.selected)
        self.e.execute(command('UNDO'))
        self.assertEqual(self.e.selected, 1)
        self.e.execute(command('CLEAR_CANVAS'))
        self.assertEqual(self.e.objects, {})
        self.e.execute(command('UNDO'))
        self.assertEqual(set(self.e.objects), {1, 2})

    def test_failed_operations_atomic(self):
        before = deepcopy((self.e.objects, self.e.selected, self.e.next_id, self.e.history))
        for c in [command('MOVE_OBJECT', id=99, dx=1, dy=1), command('SET_WIDTH', id=1, value=0),
                  command('MOVE_OBJECT', id=1, dx=1000000, dy=0)]:
            with self.assertRaises(CommandError):
                self.e.execute(c)
            self.assertEqual((self.e.objects, self.e.selected, self.e.next_id, self.e.history), before)

    def test_zero_line_and_type_error(self):
        self.e.execute(command('CREATE_LINE', x1=0, y1=0, x2=0, y2=0))
        with self.assertRaises(CommandError):
            self.e.execute(command('SET_LENGTH', id=2, value=3))
        self.e.execute(command('CREATE_POINT', x=0, y=0))
        with self.assertRaises(CommandError):
            self.e.execute(command('GET_LENGTH', id=3))

    def test_queries_do_not_change_undo(self):
        self.e.execute(command('GET_POSITION', id=1))
        self.e.execute(command('UNDO'))
        self.assertEqual(self.e.objects, {})
        self.assertEqual(self.e.next_id, 1)
        with self.assertRaises(CommandError):
            self.e.execute(command('UNDO'))

    def test_point_circle_rotation_noop(self):
        self.e.execute(command('CREATE_CIRCLE', x=1, y=2, radius=3))
        before = deepcopy(self.e.objects[2])
        self.e.execute(command('ROTATE_OBJECT', id=2, angle=90))
        self.assertEqual(self.e.objects[2], before)

class EvaluationTests(unittest.TestCase):
    def test_all_generated_commands_execute(self):
        pairs = generate_pairs(100, 8)
        setup = [command('CREATE_LINE', x1=0, y1=0, x2=6, y2=0) for _ in range(40)]
        cases = [dict(question=q, answer=a, setup=setup) for q, a in pairs]
        answers = dict(pairs)
        report = evaluate_cases(cases, answers.__getitem__)
        for metric in ('intent_accuracy', 'argument_accuracy', 'exact_command_accuracy', 'valid_command_rate', 'execution_success_rate'):
            self.assertEqual(report['overall'][metric], 1)

    def test_metrics_and_fixtures(self):
        cases = [{'question': 'q1', 'answer': canonical(command('COUNT_OBJECTS', type='line')), 'setup': []},
                 {'question': 'q2', 'answer': canonical(command('GET_POSITION', id=1)),
                  'setup': [command('CREATE_POINT', x=0, y=0)]}]
        predictions = {'q1': canonical(command('COUNT_OBJECTS', type='line')), 'q2': canonical(command('GET_POSITION', id=99))}
        report = evaluate_cases(cases, predictions.__getitem__)
        overall = report['overall']
        self.assertEqual(overall['intent_accuracy'], 1)
        self.assertEqual(overall['argument_accuracy'], .5)
        self.assertEqual(overall['exact_command_accuracy'], .5)
        self.assertEqual(overall['valid_command_rate'], 1)
        self.assertEqual(overall['execution_success_rate'], .5)
        self.assertEqual(len(report['by_command']), 20)

    def test_malformed_and_unmeasured(self):
        cases = [{'question': 'q', 'answer': canonical(command('UNDO'))}]
        report = evaluate_cases(cases, lambda q: 'oops')['overall']
        self.assertEqual(report['valid_command_rate'], 0)
        self.assertIsNone(report['execution_success_rate'])

if __name__ == '__main__':
    unittest.main()
