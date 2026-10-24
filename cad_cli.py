"""Local CAD console with explicitly separate model and executor test modes."""
import argparse
import json
from pathlib import Path
from executor import Executor
from cad_schema import parse_command, canonical

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executor-test', action='store_true', help='Direct JSON only; no Transformer predictions.')
    parser.add_argument('--checkpoint', default=str(Path(__file__).with_name('cad_model.pt')))
    args = parser.parse_args()
    predictor = None
    if not args.executor_test:
        from inference import Predictor
        try:
            predictor = Predictor(args.checkpoint)
        except (ValueError, OSError) as exc:
            parser.exit(2, str(exc) + '\n')
    print('Executor Test Mode: enter JSON; no model is used.' if args.executor_test else 'CAD Controller Ready: Transformer inference mode.')
    print('Type exit to quit.')
    executor = Executor()
    while True:
        try:
            question = input('JSON: ' if args.executor_test else 'You: ').strip()
            if question.lower() in ('exit', 'quit'):
                break
            if not question:
                continue
            text = question if args.executor_test else predictor.predict(question)
            c = parse_command(text)
            print('Command: ' + canonical(c))
            print('Result: ' + json.dumps(executor.execute(c), allow_nan=False))
        except (EOFError, KeyboardInterrupt):
            break
        except (ValueError, OSError) as exc:
            print('Error: ' + str(exc))

if __name__ == '__main__':
    main()
