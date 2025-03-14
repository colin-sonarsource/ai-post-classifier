# AI Post Classifier

This Python script classifies forum posts by leveraging OpenAI's  API. It reads a prompt from a text file and processes a CSV file containing forum posts, updating it with classification results.

## Features

- Reads a custom prompt from a text file.
- Processes CSV files with forum posts. It supports:
  - A column named `post`, or
  - Columns named `Thread Title` and `Raw Post Content` (which are combined into a single message).
- Can use multiple OpenAI models to classify posts.
- Logs progress and errors during classification.

## Prerequisites

- Python 3.6+
- The required Python packages listed in [requirements.txt](/Users/colin/Source/ai-post-classifier-validation/requirements.txt):
  - `openai`
  - `pandas`

Install the dependencies using:

```sh
pip install -r requirements.txt
```

## Usage

Run the script from the command line with the following arguments:

* prompt_file: Path to the text file containing the classification prompt.
* csv_file: Path to the CSV file containing forum posts.
* --models: List of OpenAI models to use. Provide model names separated by spaces.