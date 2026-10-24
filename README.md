# TinyCAD: Natural Language CAD Controller

A local Python prototype adapting the supplied Tiny English Chatbot. The original
`TinyChatbot` architecture is retained: learned token and positional embeddings,
PyTorch multi-head self-attention with Q/K/V projections, causal masking, GELU
feed-forward layers, layer normalization and an output projection. No pretrained
model, external LLM API, web server or CAD application is used.

**Delivery status:** implemented and tested, but no CAD training CSV or trained
CAD checkpoint is supplied. Natural-language inference requires future training;
its accuracy is not established. The original `model.pt` is preserved byte for byte
and is not a CAD checkpoint. The original archive remains unchanged.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Original causal Transformer architecture, unchanged. |
| `model.pt` | Original chatbot weights, preserved; never used by the CAD CLI. |
| `cad_schema.py` | Authoritative 20 schemas, strict JSON parser, range and semantic validation, canonical JSON serialization. |
| `command_schemas.json` | Generated reference copy of the schemas; runtime imports `cad_schema.py`. |
| `executor.py` | Transactional geometry, real-state queries, selection, duplication and undo. |
| `generate_dataset_1.py` | Seeded, bounded template/parameter sampler; importing never writes data. |
| `data_utils.py` | Lossless ASCII character tokenizer, CAD CSV validation, deduplication and conflict detection. |
| `train.py` | Answer-only cross-entropy, validation loss, early stopping, best `cad_model.pt`, reproducible splits. |
| `inference.py` | Checkpoint compatibility checks and greedy autoregressive decoding. |
| `cad_cli.py` | Local model inference or explicitly labeled direct-JSON executor testing. |
| `evaluate.py` | Five metrics overall and per command, with optional execution pre-state fixtures. |
| `tests/test_cad.py` | Schemas, in-memory generation, geometry, errors, transactions, undo and evaluation. |
| `tests/test_model.py` | Forward-only causal-mask, context, target-masking and split tests; no training. |
| `requirements.txt` | Runtime dependency; Python 3.8+ with compatible PyTorch. |
| `VALIDATION.md` | Delivery verification results and known limitations. |

The unpacked original project is retained in the local
`B5c_TinyLLM_3_MiniChatbot/` backup folder. It is excluded from the deliverable ZIP
because it contains old chat datasets, web assets, repository metadata and caches.
All runnable CAD project files are at the top level of `B5d_LLMUpdateToCAD/`.

## Setup and immediate executor testing

```powershell
cd S:\CodexFiles\B5d_LLMUpdateToCAD
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python cad_cli.py --executor-test
```

Enter these JSON commands one per prompt:

```json
{"command":"CREATE_LINE","x1":0,"y1":0,"x2":6,"y2":0}
{"command":"COUNT_OBJECTS","type":"line"}
{"command":"SET_LENGTH","id":1,"value":8}
{"command":"GET_LENGTH","id":1}
{"command":"UNDO"}
```

The results include a created line, count `1`, then length `8`. `UNDO` reverses
the length change, since queries do not create history entries. Type `exit` to quit.
Test mode parses JSON directly and never claims to predict natural language.

## Fixed command contract

Every field shown is required; no extra fields are accepted. All commands also
require `command` containing the exact uppercase name. Refer to
`command_schemas.json` or `cad_schema.SCHEMAS` for machine-readable definitions.

| Command | Other fields |
| --- | --- |
| `CREATE_LINE` | `x1,y1,x2,y2`: numbers |
| `CREATE_RECTANGLE` | `x,y`: anchor; `width,height`: positive geometric dimensions |
| `CREATE_CIRCLE` | `x,y`: center; `radius`: positive |
| `CREATE_POINT` | `x,y`: position |
| `SET_LENGTH` | `id`: integer; `value`: positive line length |
| `MOVE_OBJECT` | `id,dx,dy`: relative displacement |
| `ROTATE_OBJECT` | `id,angle`: signed degrees about anchor |
| `SET_COLOR` | `id,color` |
| `SET_WIDTH` | `id,value`: positive stroke width |
| `DELETE_OBJECT` | `id` |
| `COUNT_OBJECTS` | `type` |
| `COUNT_IF` | `type,property,operator,value` |
| `FIND_OBJECTS` | `type,property,operator,value`; returns sorted IDs |
| `GET_LENGTH` | `id`; line only |
| `GET_POSITION` | `id`; returns anchor `x,y` |
| `GET_PROPERTIES` | `id`; returns a detached properties dictionary |
| `SELECT_OBJECT` | `id` |
| `DUPLICATE_OBJECT` | `id,dx,dy` |
| `CLEAR_CANVAS` | none |
| `UNDO` | none |

- Types: `line`, `rectangle`, `circle`, `point`; queries also accept `all`.
- Colors: black, white, red, green, blue, yellow, cyan, magenta.
- Condition properties: `length`, `radius`, `width` (stroke width), `color`, `x`, `y`.
- Operators: `eq`, `ne`, `lt`, `le`, `gt`, `ge`; color permits `eq`/`ne` only.
- Numeric parameters are finite and bounded by magnitude 1,000,000. Dimensions
  and widths must be positive. IDs are integers 1 through 2,147,483,647; booleans
  are rejected. Duplicate JSON keys, missing keys and extra keys are rejected.
- Numeric equality uses `math.isclose(rel_tol=1e-9, abs_tol=1e-9)`; strict ordered
  comparisons exclude values considered equal under this tolerance.
- Conditions on unavailable properties do not match an object. `length` applies
  only to lines, and `radius` only to circles. `x,y` refer to the anchor.
- Creation defaults to black and stroke width 1. The rectangle's creation `width`
  becomes `geometry_width` in stored properties; stored `width` is stroke width.

## Geometry and state semantics

Coordinates use a Cartesian plane with positive y upward; units are abstract.
Lines store two endpoints; their anchor is the first endpoint. `SET_LENGTH` fixes
that endpoint and preserves direction. Zero-length lines may be created, but
cannot have their length changed because their direction is undefined.

Rectangles store anchor `x,y`, `geometry_width`, `height` and an orientation
`angle` in degrees. Their corners are anchor plus rotated local offsets
`(0,0)`, `(geometry_width,0)`, `(geometry_width,height)`, `(0,height)`.
Rotation is counterclockwise for positive angles, about the anchor. A line's
second endpoint rotates about its first. A rectangle's orientation changes
modulo 360. A circle or point rotated about its own anchor is unchanged.

Move and duplicate use relative `dx,dy`, not an absolute destination. Duplicate
copies all geometry and style into a new ID; selection stays unchanged. IDs are
monotonic during normal operations, including clear/delete. Undo restores the
entire previous state, including selection and ID allocation, so undo can reuse
an ID. Selection changes are undoable. No-op commands and queries do not create
history. Undo itself does not create redo history. Failed operations roll back
without changing history. This is an in-memory prototype; history is unbounded
and the session is not persisted.

## Next: Dataset 1 and Dataset 2

These commands are instructions for future use; they were **not run** during
development. The generator refuses to overwrite existing files. It samples
instead of taking a Cartesian product, enforces complete command coverage,
deduplicates inputs and validates every label against the shared contract.
Finite commands such as UNDO have few unique expressions, so the remaining
quota is redistributed; class counts are not exactly balanced.

```powershell
python generate_dataset_1.py --count 10000 --seed 42 --split train --output dataset_1.csv
python generate_dataset_1.py --count 1000 --seed 43 --split validation --output validation.csv
python generate_dataset_1.py --count 1000 --seed 44 --split test --output test.csv
```

`PREFIXES`, `POOLS`, `ID_POOLS` and `sample()` are extension points. Each command
has multiple polite/contextual templates. Validation and test use unseen
prefixes, disjoint numeric pools and disjoint ID pools. They share the underlying
command sentence structure and categorical palettes; this is a limited
generalization test, not proof of broad natural-language understanding. Extend
with entirely new sentence structures and manually reviewed cases before
claiming broad generalization. Parameter-free commands necessarily share labels
across these explicit splits, but their questions differ.

Prepare Dataset 2 manually or with a separate future process. No AI API client
or content is provided. Its CSV header must be exactly `question,answer`.
Use CSV quoting for JSON (a CSV writer is recommended), English printable ASCII
questions and valid JSON answers matching the schemas. Include diverse natural
phrasing, explicit IDs and unambiguous parameters; exclude multi-command requests
and implicit references such as "it" because this model sees no conversation or
canvas context. Keep held-out Dataset 2 paraphrases and parameter combinations
out of training. A repeated normalized question with conflicting labels is an
error; identical pairs are deduplicated across merged datasets.

## Next: training

```powershell
python train.py --data dataset_1.csv dataset_2.csv --validation validation.csv --test test.csv --max-length 384
```

Missing optional input files are skipped; if both CAD datasets are absent,
training exits with an explanatory error and creates no output. Legacy chatbot
datasets are never loaded. If `--validation` and `--test` are omitted, the merged
data is split 80/10/10 by distinct canonical JSON labels, keeping all paraphrases
of one label in the same partition. This is a fallback parameter holdout, not an
unseen-template benchmark. Explicit partitions are recommended; they are checked
for normalized question overlap. Review per-command coverage in held-out data,
especially small parameter-free categories.

The ASCII character tokenizer has a fixed vocabulary of 100 entries including
five special tokens, preserves JSON case/punctuation and signed decimal digits,
and handles unseen numeric strings without unknown number tokens. Non-ASCII or
control characters are rejected. Character sequences are longer than word-token
sequences; examples exceeding context fail explicitly rather than being dropped.

Training masks question and padding targets, optimizes answer/EOS cross-entropy
with AdamW and gradient clipping, monitors validation loss and saves the best
checkpoint with early stopping. The test set never selects model weights.
Outputs are `cad_model.pt`, `cad_model.loss.json` and `cad_splits.json` (the latter
contains the actual held-out pairs). `model.pt` is protected. Checkpoints include
schema/tokenizer versions, vocabulary, configuration and split fingerprints.
Do not change schemas without incrementing the schema version and retraining.

## Next: model inference and evaluation

```powershell
python cad_cli.py
python evaluate.py --checkpoint cad_model.pt --split-file cad_splits.json --output evaluation.json
python evaluate.py --checkpoint cad_model.pt --data test.csv --output heldout_evaluation.json
```

The CLI generates characters autoregressively from the trained Transformer.
There is no hand-written natural-language parser, repair heuristic or fallback.
Malformed JSON, invalid arguments, missing objects and incomplete generations
produce an error. Schema-valid output can still express the wrong user intent;
first-version training and evaluation must establish its quality.

Evaluation reports all 20 command classes (null metrics for zero examples):

1. Intent Accuracy: correct command name / all examples.
2. Argument Accuracy: exact equality of all non-command fields / all examples,
   independently of intent. No extra or missing argument is allowed.
3. Exact Command Accuracy: valid prediction identical to the entire gold command
   / all examples; JSON key order and whitespace do not matter.
4. Valid Command Rate: strict parse and shared-schema validation / all examples.
5. Execution Success Rate: exception-free predicted execution / fixture examples,
   irrespective of intent correctness. It is **null** without explicit pre-state
   fixtures, since a standalone question/answer CSV cannot specify real state.

For execution measurement use `--cases execution_cases.json`, with a JSON list
of objects containing `question`, `answer` (a JSON string), and `setup` (a list
of schema-valid command dictionaries). Each case starts with a fresh executor;
setup is replayed for gold and prediction, and invalid gold setups fail the
evaluation. An UNDO fixture needs at least one preceding state change. Example:

```json
[
  {
    "question": "Change line 1 to length 8.",
    "answer": "{\"command\":\"SET_LENGTH\",\"id\":1,\"value\":8}",
    "setup": [{"command":"CREATE_LINE","x1":0,"y1":0,"x2":6,"y2":0}]
  }
]
```

The evaluator rejects questions seen in checkpoint training/validation metadata.
For intent/argument metrics it also inspects parseable but schema-invalid output;
validity and exact-command metrics still require strict validation. It reports
execution feasibility, not final-state semantic equivalence.
