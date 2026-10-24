"""Transactional in-memory 2D geometry. No language interpretation occurs here."""
import math
from copy import deepcopy
from cad_schema import CommandError, parse_command, validate_command

class Executor:
    def __init__(self):
        self.objects = {}
        self.selected = None
        self.next_id = 1
        self.history = []

    def _snapshot(self):
        return deepcopy((self.objects, self.selected, self.next_id))

    def execute(self, command):
        c = parse_command(command) if isinstance(command, str) else validate_command(deepcopy(command))
        name = c['command']
        if name == 'UNDO':
            if not self.history:
                raise CommandError('Nothing to undo.')
            self.objects, self.selected, self.next_id = self.history.pop()
            return 'Undone.'
        before = self._snapshot()
        try:
            result = self._apply(c)
            # Reject derived geometry overflow and roll back without a history entry.
            for obj in self.objects.values():
                for key, value in obj.items():
                    if key != 'id' and type(value) in (int, float) and (not math.isfinite(value) or abs(value) > 1000000):
                        raise CommandError('Resulting geometry exceeds coordinate limits.')
        except Exception:
            self.objects, self.selected, self.next_id = before
            raise
        if self._snapshot() != before:
            self.history.append(before)
        return deepcopy(result)

    def _get(self, id):
        if id not in self.objects:
            raise CommandError('Object does not exist: ' + str(id))
        return self.objects[id]

    @staticmethod
    def length(obj):
        if obj['type'] != 'line':
            raise CommandError('Length is defined only for lines.')
        return math.hypot(obj['x2'] - obj['x1'], obj['y2'] - obj['y1'])

    @staticmethod
    def position(obj):
        return {'x': obj['x1'], 'y': obj['y1']} if obj['type'] == 'line' else {'x': obj['x'], 'y': obj['y']}

    @staticmethod
    def move(obj, dx, dy):
        for x, y in [('x1', 'y1'), ('x2', 'y2')] if obj['type'] == 'line' else [('x', 'y')]:
            obj[x] += dx
            obj[y] += dy

    def _matches(self, obj, c):
        if c['type'] != 'all' and obj['type'] != c['type']:
            return False
        if 'property' not in c:
            return True
        prop = c['property']
        if prop == 'length':
            if obj['type'] != 'line':
                return False
            actual = self.length(obj)
        elif prop in ('x', 'y'):
            actual = self.position(obj)[prop]
        else:
            if prop not in obj:
                return False
            actual = obj[prop]
        value, op = c['value'], c['operator']
        eq = math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-9) if type(actual) in (int, float) else actual == value
        return {'eq': lambda: eq, 'ne': lambda: not eq, 'lt': lambda: actual < value and not eq,
                'le': lambda: actual < value or eq, 'gt': lambda: actual > value and not eq,
                'ge': lambda: actual > value or eq}[op]()

    def _apply(self, c):
        name = c['command']
        if name.startswith('CREATE_'):
            obj = {k: v for k, v in c.items() if k != 'command'}
            obj.update(id=self.next_id, type=name[7:].lower(), color='black')
            if obj['type'] == 'rectangle':
                obj['geometry_width'] = obj.pop('width')
                obj['angle'] = 0
            obj['width'] = 1
            self.objects[self.next_id] = obj
            self.next_id += 1
            return 'Created %s %s.' % (obj['type'], obj['id'])
        if name in ('COUNT_OBJECTS', 'COUNT_IF', 'FIND_OBJECTS'):
            ids = [id for id, obj in sorted(self.objects.items()) if self._matches(obj, c)]
            return ids if name == 'FIND_OBJECTS' else len(ids)
        if name == 'CLEAR_CANVAS':
            self.objects.clear()
            self.selected = None
            return 'Canvas cleared.'
        obj = self._get(c['id'])
        if name == 'GET_LENGTH':
            return self.length(obj)
        if name == 'GET_POSITION':
            return self.position(obj)
        if name == 'GET_PROPERTIES':
            return obj
        if name == 'SELECT_OBJECT':
            self.selected = obj['id']
        elif name == 'DELETE_OBJECT':
            del self.objects[obj['id']]
            if self.selected == obj['id']:
                self.selected = None
        elif name == 'SET_COLOR':
            obj['color'] = c['color']
        elif name == 'SET_WIDTH':
            obj['width'] = c['value']
        elif name == 'SET_LENGTH':
            length = self.length(obj)
            if length == 0:
                raise CommandError('A zero-length line has no direction.')
            scale = c['value'] / length
            obj['x2'] = obj['x1'] + (obj['x2'] - obj['x1']) * scale
            obj['y2'] = obj['y1'] + (obj['y2'] - obj['y1']) * scale
        elif name == 'MOVE_OBJECT':
            self.move(obj, c['dx'], c['dy'])
        elif name == 'ROTATE_OBJECT':
            if obj['type'] == 'line':
                theta = math.radians(c['angle'])
                dx, dy = obj['x2'] - obj['x1'], obj['y2'] - obj['y1']
                obj['x2'] = obj['x1'] + dx * math.cos(theta) - dy * math.sin(theta)
                obj['y2'] = obj['y1'] + dx * math.sin(theta) + dy * math.cos(theta)
            elif obj['type'] == 'rectangle':
                obj['angle'] = (obj['angle'] + c['angle']) % 360
            # A circle or point rotated about its own anchor is unchanged.
        elif name == 'DUPLICATE_OBJECT':
            obj = deepcopy(obj)
            obj['id'] = self.next_id
            self.move(obj, c['dx'], c['dy'])
            self.objects[self.next_id] = obj
            self.next_id += 1
            return 'Created duplicate %s.' % obj['id']
        return name + ' completed.'
