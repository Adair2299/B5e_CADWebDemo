# Delivery validation

Validated on Python 3.8.10 and PyTorch 2.4.1+cpu. No optimizer steps, model
training, formal dataset generation or web development were performed.

- `python -m unittest discover -s tests -v`: **22 tests passed**.
- Generated train/validation/test samples only in memory; checked reproducibility,
  all 20 commands, unique questions and strict schema validation.
- Executed 100 generated command cases against independent in-memory fixtures;
  every command category executed successfully. This is an executor test using
  gold labels, not a trained model accuracy claim.
- Tested bad JSON, duplicated keys, booleans as IDs, invalid dimensions/colors,
  missing objects, derived coordinate overflow and transactional rollback.
- Tested line lengths, translation, rotation, rectangle dimensions versus stroke
  width, colors, query comparisons, duplication, selection, deletion, clear and undo.
- Tested CSV parsing/deduplication/conflict rejection through in-memory mocks;
  no test CSVs were saved.
- Forward-only tests verified causal masking, output shape, context limits,
  answer-only targets and ignored padding. Original architecture preserved.
- Verified the CLI sequence create/count/set length/query/undo/query returns
  count 1, length 8 and restored length 6.
- Verified `python train.py` refuses absent CAD data and `python cad_cli.py`
  refuses an absent CAD checkpoint. Neither creates a model or dataset.
- Verified original `model.py` and `model.pt` against members of the original
  ZIP. Original checkpoint SHA-256:
  `ae23bb479e6fa3b5255d9656d26c2ecf62b00b9cdc1fdb5465ca7c4255b234d0`.
- Original ZIP SHA-256:
  `06500170592a1a9fdb3596c1c533f82913db8d4756dbf64d59996798e66fb55c`.
- Deliverable ZIP excludes the unpacked legacy project, caches, old CSVs,
  repository metadata and web assets. It includes the unchanged original weights.

## Limits

No trained CAD model exists yet, so model intent, argument and execution quality
remain unmeasured. Character tokenization increases context length. The baseline
Transformer is small and may require tuning and more diverse reviewed examples.
Held-out template prefixes do not replace a broad independent natural-language
benchmark. This prototype handles one explicit-ID command at a time and has no
dialogue context, rendering, disk-persisted canvas or real CAD integration.
