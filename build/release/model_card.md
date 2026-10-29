---
base_model: $base
base_model_relation: adapter
library_name: peft
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - lora
  - plantuml
  - uml
  - diagram-recognition
  - image-to-code
---

# $base, fine-tuned for image-to-PlantUML (LoRA)

LoRA $adapter_word for [$base](https://huggingface.co/$base) that $verb a rendered UML class or
sequence diagram into its PlantUML source. $They $are the fine-tuned $model_word of the study
[Fine-tuning for image-to-PlantUML: Qwen3.5 at 2B, 9B and 27B]($study_url), which runs the
[image-to-PlantUML benchmark]($benchmark_url) on the base model before and after fine-tuning.

## Adapters

| Subfolder | Training pairs | Checkpoint | sha256 of `adapter_model.safetensors` |
|---|---|---|---|
$adapter_rows

Each subfolder holds `adapter_model.safetensors` and `adapter_config.json`, byte for byte the files
the study served for its test-set runs.

## Base model

`$base` at revision `$revision`. Training and evaluation used this revision.
`adapter_config.json` does not record it, so pass it when you load the base.

## Loading

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

base = AutoModelForImageTextToText.from_pretrained(
    "$base", revision="$revision", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "$repo_id", subfolder="$first_subfolder").eval()
processor = AutoProcessor.from_pretrained("$base", revision="$revision")
```

## Input

The study trained and evaluated on this input. Behaviour outside it is undefined.

- One user turn: the picture, one newline, then the benchmark's prompt
  ([`prompts/zero_shot.txt`]($benchmark_url/blob/main/prompts/zero_shot.txt)). The study builds the
  turn with `build/user_turn.py` of its repository.
- The picture: a PlantUML class or sequence diagram, rendered and standardized as the benchmark's
  test images (`data/puml_images_1568/`).
- The model's own chat template, with a generation prompt and `enable_thinking=False`.
- Greedy decoding, up to 5,376 new tokens.

## Training

The training (the data, the recipe, how the checkpoint was chosen) and the scores on the benchmark
are described in the study repository: the
[training recipe]($study_url/blob/main/methodology/training-recipe.md), the
[training set]($study_url/blob/main/methodology/training-set-construction.md), and the test-set
runs and tables in `runs.zip`.

## Licence

Apache-2.0, the licence of the base model (see `LICENSE`). The training diagrams were mined from
open-source repositories, and each remains under the licence of the repository it came from; they
are not distributed here.

## Citation

See [`CITATION.cff`]($study_url/blob/main/CITATION.cff) in the study repository.
