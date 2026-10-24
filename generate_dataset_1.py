"""Bounded, balanced CAD data generation. Importing never writes a dataset."""
import argparse
import csv
import random
from cad_schema import DEFINITIONS, TYPES, COLORS, OPS, canonical

PREFIXES = {
    'train': ['', 'Please ', 'Could you '],
    'validation': ['I would like you to ', 'For this drawing, '],
    'test': ['Would you kindly ', 'In my CAD workspace, '],
}
POOLS = {'train': [-10, -3, 0, 1, 2, 4, 6, 10],
         'validation': [-7.5, -1.5, 0.5, 3.5, 8.5],
         'test': [-12.25, -2.25, 1.25, 5.25, 11.25]}
ID_POOLS = {'train': list(range(1, 21)), 'validation': list(range(21, 31)), 'test': list(range(31, 41))}
COMPARISONS = {'eq': 'equal to', 'ne': 'not equal to', 'lt': 'less than',
               'le': 'at most', 'gt': 'greater than', 'ge': 'at least'}
ALTERNATES = {
    'CREATE_LINE': 'connect ({x1},{y1}) and ({x2},{y2}) with a line.',
    'CREATE_RECTANGLE': 'add a rectangle of width {width} and height {height} anchored at ({x},{y}).',
    'CREATE_CIRCLE': 'add a circle of radius {radius} at center ({x},{y}).',
    'CREATE_POINT': 'mark the coordinates ({x},{y}) with a point.',
    'SET_LENGTH': 'make line {id} have length {value}.',
    'SET_WIDTH': 'give object {id} a stroke width of {value}.',
    'MOVE_OBJECT': 'translate object {id} by ({dx},{dy}).',
    'ROTATE_OBJECT': 'turn object {id} through {angle} degrees counterclockwise around its anchor.',
    'SET_COLOR': 'make the color of object {id} {color}.',
    'DELETE_OBJECT': 'remove object {id} from the canvas.',
    'COUNT_OBJECTS': 'tell me the number of objects with type {type}.',
    'GET_LENGTH': 'what is the length of line {id}?',
    'GET_POSITION': 'where is the anchor of object {id}?',
    'GET_PROPERTIES': 'what are the properties of object {id}?',
    'SELECT_OBJECT': 'make object {id} the selected object.',
    'DUPLICATE_OBJECT': 'copy object {id} with offset ({dx},{dy}).',
    'CLEAR_CANVAS': 'remove every object from the drawing.',
    'UNDO': 'reverse the most recent state change.',
}

def sample(name, rng, split):
    number = lambda: rng.choice(POOLS[split])
    positive = lambda: rng.choice([n for n in POOLS[split] if n > 0])
    c = {'command': name}
    if 'id' in DEFINITIONS[name]:
        c['id'] = rng.choice(ID_POOLS[split])
    if name == 'CREATE_LINE':
        c.update(x1=number(), y1=number(), x2=number(), y2=number())
        q = 'draw a line from ({x1},{y1}) to ({x2},{y2}).'
    elif name == 'CREATE_RECTANGLE':
        c.update(x=number(), y=number(), width=positive(), height=positive())
        q = 'create a rectangle anchored at ({x},{y}) with width {width} and height {height}.'
    elif name == 'CREATE_CIRCLE':
        c.update(x=number(), y=number(), radius=positive())
        q = 'draw a circle centered at ({x},{y}) with radius {radius}.'
    elif name == 'CREATE_POINT':
        c.update(x=number(), y=number())
        q = 'place a point at ({x},{y}).'
    elif name in ('SET_LENGTH', 'SET_WIDTH'):
        c['value'] = positive()
        q = 'set the ' + ('length' if name == 'SET_LENGTH' else 'stroke width') + ' of object {id} to {value}.'
    elif name in ('MOVE_OBJECT', 'DUPLICATE_OBJECT'):
        c.update(dx=number(), dy=number())
        q = ('move object' if name == 'MOVE_OBJECT' else 'duplicate object') + ' {id} by dx {dx} and dy {dy}.'
    elif name == 'ROTATE_OBJECT':
        c['angle'] = number()
        q = 'rotate object {id} by {angle} degrees counterclockwise about its anchor.'
    elif name == 'SET_COLOR':
        c['color'] = rng.choice(COLORS)
        q = 'change object {id} to {color}.'
    elif name == 'COUNT_OBJECTS':
        c['type'] = rng.choice(TYPES + ['all'])
        q = 'count objects of type {type}.'
    elif name in ('COUNT_IF', 'FIND_OBJECTS'):
        c['type'] = rng.choice(TYPES + ['all'])
        props = ['width', 'color', 'x', 'y']
        if c['type'] in ('line', 'all'):
            props.append('length')
        if c['type'] in ('circle', 'all'):
            props.append('radius')
        c['property'] = rng.choice(props)
        c['operator'] = rng.choice(['eq', 'ne'] if c['property'] == 'color' else OPS)
        c['value'] = rng.choice(COLORS) if c['property'] == 'color' else number()
        q = ('count' if name == 'COUNT_IF' else 'find') + ' objects of type {type} whose {property} is ' + COMPARISONS[c['operator']] + ' {value}.'
    else:
        q = {'DELETE_OBJECT': 'delete object {id}.', 'GET_LENGTH': 'report the length of line {id}.',
             'GET_POSITION': 'report the anchor position of object {id}.',
             'GET_PROPERTIES': 'show all properties of object {id}.',
             'SELECT_OBJECT': 'select object {id}.', 'CLEAR_CANVAS': 'clear the canvas.',
             'UNDO': 'undo the last state change.'}[name]
    if name in ('COUNT_IF', 'FIND_OBJECTS'):
        alternate = ('how many' if name == 'COUNT_IF' else 'list the IDs of') + ' objects of type {type} have {property} ' + COMPARISONS[c['operator']] + ' {value}?'
    else:
        alternate = ALTERNATES[name]
    q = rng.choice([q, alternate])
    # Multiple surface templates for every command; held-out prefixes never enter training.
    return rng.choice(PREFIXES[split]) + q.format(**c), canonical(c)

def generate_pairs(count=2000, seed=42, split='train'):
    if count < 20:
        raise ValueError('count must be at least 20 to cover all commands.')
    if split not in PREFIXES:
        raise ValueError('Unknown split.')
    rng = random.Random(seed)
    names = list(DEFINITIONS)
    pairs, seen = [], {}
    # Small finite commands such as UNDO naturally exhaust; redistribute their quota.
    for name in names:
        q, a = sample(name, rng, split)
        seen[q.lower()] = a
        pairs.append((q, a))
    attempts = 0
    while len(pairs) < count:
        attempts += 1
        if attempts > count * 200:
            raise ValueError('Requested count exceeds bounded template space. Extend pools or lower count.')
        q, a = sample(names[(attempts - 1) % len(names)], rng, split)
        key = q.lower()
        if key in seen:
            if seen[key] != a:
                raise ValueError('Conflicting template labels.')
            continue
        seen[key] = a
        pairs.append((q, a))
    rng.shuffle(pairs)
    return pairs

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', choices=list(PREFIXES), default='train')
    parser.add_argument('--output', default='dataset_1.csv')
    args = parser.parse_args()
    pairs = generate_pairs(args.count, args.seed, args.split)
    # Refuse accidental overwrites, especially of held-out evaluation data.
    with open(args.output, 'x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['question', 'answer'])
        writer.writerows(pairs)
    print('Wrote %s validated pairs to %s.' % (len(pairs), args.output))

if __name__ == '__main__':
    main()
