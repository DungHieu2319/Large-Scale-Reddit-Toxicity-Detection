from datasets import load_dataset
import pandas as pd

def load_reddit_sample(limit=50000):

    print("🚀 Loading dataset (streaming)...")

    dataset = load_dataset(
        "fddemarco/pushshift-reddit-comments",
        split="train",
        streaming=True
    )

    data = []

    for i, item in enumerate(dataset):
        if item["body"] is not None:
            data.append({"body": item["body"]})

        if i >= limit:
            break

        if i % 10000 == 0:
            print(f"Loaded {i} samples...")

    df = pd.DataFrame(data)

    # lưu file
    df.to_csv("data/raw/reddit_sample.csv", index=False)

    print("✅ DONE! Saved to data/raw/reddit_sample.csv")

if __name__ == "__main__":
    load_reddit_sample(50000)