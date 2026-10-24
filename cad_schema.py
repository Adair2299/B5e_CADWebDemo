"""Single source of truth: all generators, inference and execution use SCHEMAS."""
import json
import math
from copy import deepcopy

VERSION = 1
TYPES = ['line', 'rectangle', 'circle', 'point']
COLORS = ['black', 'white', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
PROPERTIES = ['length', 'radius', 'width', 'color', 'x', 'y']
OPS = ['eq', 'ne', 'lt', 'le', 'gt', 'ge']
NUM = {'type': 'number', 'minimum': -1000000, 'maximum': 1000000}
POS = {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 1000000}
ID = {'type': 'integer', 'minimum': 1, 'maximum': 2147483647}
def enum(values):
    return {'type': 'string', 'enum': values}
def fields(**kwargs):
    return kwargs
DEFINITIONS = {
    'CREATE_LINE': fields(x1=NUM, y1=NUM, x2=NUM, y2=NUM),
    'CREATE_RECTANGLE': fields(x=NUM, y=NUM, width=POS, height=POS),
    'CREATE_CIRCLE': fields(x=NUM, y=NUM, radius=POS),
    'CREATE_POINT': fields(x=NUM, y=NUM),
    'SET_LENGTH': fields(id=ID, value=POS),
    'MOVE_OBJECT': fields(id=ID, dx=NUM, dy=NUM),
    'ROTATE_OBJECT': fields(id=ID, angle=NUM),
    'SET_COLOR': fields(id=ID, color=enum(COLORS)),
    'SET_WIDTH': fields(id=ID, value=POS),
    'DELETE_OBJECT': fields(id=ID),
    'COUNT_OBJECTS': fields(type=enum(TYPES + ['all'])),
    'COUNT_IF': fields(type=enum(TYPES + ['all']), property=enum(PROPERTIES), operator=enum(OPS), value={'type': ['number', 'string']}),
    'FIND_OBJECTS': fields(type=enum(TYPES + ['all']), property=enum(PROPERTIES), operator=enum(OPS), value={'type': ['number', 'string']}),
    'GET_LENGTH': fields(id=ID), 'GET_POSITION': fields(id=ID),
    'GET_PROPERTIES': fields(id=ID), 'SELECT_OBJECT': fields(id=ID),
    'DUPLICATE_OBJECT': fields(id=ID, dx=NUM, dy=NUM),
    'CLEAR_CANVAS': {}, 'UNDO': {},
}
SCHEMAS = {name: {'$schema': 'https://json-schema.org/draft/2020-12/schema',
    'type': 'object', 'properties': dict(command={'const': name}, **deepcopy(spec)),
    'required': ['command'] + list(spec), 'additionalProperties': False}
    for name, spec in DEFINITIONS.items()}
for _name in ('COUNT_IF', 'FIND_OBJECTS'):
    SCHEMAS[_name]['allOf'] = [{
        'if': {'properties': {'property': {'const': 'color'}}},
        'then': {'properties': {'operator': enum(['eq', 'ne']), 'value': enum(COLORS)}},
        'else': {'properties': {'value': deepcopy(NUM)}}}]

class CommandError(ValueError):
    pass

def validate_command(command):
    """Validate the fixed schemas plus cross-field semantic constraints."""
    if not isinstance(command, dict) or not isinstance(command.get('command'), str) or command['command'] not in SCHEMAS:
        raise CommandError('Unknown or missing command.')
    schema = SCHEMAS[command['command']]
    if set(command) != set(schema['required']):
        raise CommandError('Missing or extra fields.')
    for key, spec in schema['properties'].items():
        value = command[key]
        if 'const' in spec:
            continue
        kind = spec['type']
        kinds = kind if isinstance(kind, list) else [kind]
        valid = ('string' in kinds and isinstance(value, str)) or ('integer' in kinds and type(value) is int) or ('number' in kinds and type(value) in (int, float))
        if not valid:
            raise CommandError('Invalid type for ' + key)
        if type(value) in (int, float):
            if abs(value) > spec.get('maximum', 1000000) or value < spec.get('minimum', -1000000) or value <= spec.get('exclusiveMinimum', -float('inf')) or not math.isfinite(value):
                raise CommandError('Out of range: ' + key)
        if 'enum' in spec and value not in spec['enum']:
            raise CommandError('Unsupported value for ' + key)
    if command['command'] in ('COUNT_IF', 'FIND_OBJECTS'):
        if command['property'] == 'color':
            if command['value'] not in COLORS or command['operator'] not in ('eq', 'ne'):
                raise CommandError('Color conditions require a palette color and eq/ne.')
        elif type(command['value']) not in (int, float):
            raise CommandError('Numeric property requires a numeric value.')
    return command

def parse_command(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CommandError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    try:
        return validate_command(json.loads(text, object_pairs_hook=unique))
    except (ValueError, TypeError, OverflowError) as exc:
        raise CommandError(str(exc)) from exc

def canonical(command):
    return json.dumps(validate_command(command), sort_keys=True, separators=(',', ':'), allow_nan=False)
