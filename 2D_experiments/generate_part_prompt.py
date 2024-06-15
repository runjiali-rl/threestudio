from openai import OpenAI
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str, default="2D_experiments/meta_part_generation_prompt.txt")
    parser.add_argument("--output_path", type=str, default="2D_experiments/part_generation_prompt.txt")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o")
    return parser.parse_args()

def main():
    args = parse_args()
    prompt_path = args.prompt_path
    with open(prompt_path, "r") as f:
        prompt = f.read()

    client = OpenAI(
        # This is the default and can be omitted
        api_key=args.api_key,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=args.model,
    )
    output = chat_completion.choices[0].message.content
    save_path = args.output_path
    print(output)
    with open(save_path, "w") as f:
        f.write(output)
    print(f"Prompt saved to {save_path}")


if __name__ == "__main__":
    main()