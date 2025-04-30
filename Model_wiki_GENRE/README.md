# GENRE Entity Linking Implementation for Colab

This notebook provides a minimal implementation of Facebook AI's GENRE (Generative ENtity REtrieval) for entity linking in Google Colab. GENRE is a powerful sequence-to-sequence approach for entity linking that leverages generative models to directly predict entity names.

## Overview

Entity linking is the task of connecting mentions of entities in text to their corresponding entries in a knowledge base (like Wikipedia). This notebook implements the GENRE approach, which uses a generative model to predict entity names directly from text.

## Features

- Complete setup for running GENRE in Google Colab
- Implementation of Facebook AI's entity linking model
- Example code for processing text with entity mentions
- Uses prefix tree (trie) for constraining generation to valid entity names

## Getting Started

The notebook provides all necessary steps to:

1. Clone the GENRE repository
2. Set up the required fairseq dependency
3. Download the pre-trained model and prefix tree data
4. Configure the model for entity linking
5. Run example predictions

## Usage Example

```python
from genre.fairseq_model import GENRE
model = GENRE.from_pretrained("/content/models/fairseq_entity_disambiguation_aidayago").eval()

# Example text with entity mentions
text = ["Experiments associated with the [START_ENT] INS gene [END_ENT] or in very special cases other sources."]

# Generate entity links
results = model.sample(
    sentences=text,
    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
)
```

## Alternative Huggingface Implementation

The notebook also includes code for using GENRE through Huggingface's transformers library as an alternative approach.

## Dependencies

- fairseq (custom branch: fixing_prefix_allowed_tokens_fn)
- GENRE
- jsonlines
- pickle
- transformers (optional, for alternative implementation)

## References

- [GENRE GitHub Repository](https://github.com/facebookresearch/GENRE)
- Original implementation date: October 27, 2022
