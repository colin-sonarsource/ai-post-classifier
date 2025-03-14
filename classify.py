from openai import OpenAI
import pandas as pd
import argparse
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

client = OpenAI()

def load_prompt(prompt_file):
    """Read and return the prompt from a text file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    return prompt

def classify_post(model, prompt, post):
    """
    Send the prompt plus the post to the OpenAI API using the given model
    and return the classification result.
    """
    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": post}
        ],
    }
    try:
        response = client.chat.completions.create(**request_payload)
        result = response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error classifying post with model {model}: {e}")
        result = "error"
    return result

def process_posts(csv_file, prompt, models):
    """
    Load the CSV, process each forum post, and update it with OpenAI's classification for each model.
    
    The CSV can either use a 'post' column or it may contain both 'Thread Title' and 'Raw Post Content'
    (in which case, they are combined into one message starting with "title: ").
    
    A new column 'openai_classification_<model>' is added for each provided model.
    """
    df = pd.read_csv(csv_file)

    # Determine how to get post text from CSV
    if "Thread Title" in df.columns and "Raw Post Content" in df.columns:
        get_post_text = lambda row: f"title: {row['Thread Title']}\n{row['Raw Post Content']}"
    elif "post" in df.columns:
        get_post_text = lambda row: row["post"]
    else:
        raise ValueError("CSV must contain either 'post' or both 'Thread Title' and 'Raw Post Content' columns.")

    # Process posts for each model
    for model in models:
        classifications = []
        for idx, row in df.iterrows():
            post_text = get_post_text(row)
            logging.info(f"Classifying post {idx+1}/{len(df)} with model {model}")
            classification = classify_post(model, prompt, post_text)
            classifications.append(classification)
            # Optional: add a short delay to avoid rate limits
            time.sleep(1)
        df[f'openai_classification_{model}'] = classifications

    df.to_csv(csv_file, index=False)
    logging.info(f"Updated CSV saved to {csv_file}")

def main():
    parser = argparse.ArgumentParser(description="Triage community forum posts using OpenAI chat models.")
    parser.add_argument("prompt_file", help="Path to the text file containing the prompt.")
    parser.add_argument("csv_file", help="Path to the CSV file with forum posts and manual dispositions.")
    parser.add_argument("--models", nargs="+", required=True, help="List of models to use (e.g. gpt-4o-2024-08-06 o3-mini)")
    args = parser.parse_args()

    prompt = load_prompt(args.prompt_file)
    process_posts(args.csv_file, prompt, args.models)

if __name__ == "__main__":
    main()
