import requests
import os

# tạo folder nếu chưa có
os.makedirs("data/raw", exist_ok=True)

urls = [
    "https://huggingface.co/datasets/fddemarco/pushshift-reddit-comments/resolve/main/data/RC_2015-01.parquet",
    "https://huggingface.co/datasets/fddemarco/pushshift-reddit-comments/resolve/main/data/RC_2015-02.parquet",
]

def download_file(url):
    filename = url.split("/")[-1]
    filepath = os.path.join("data/raw", filename)

    print(f"⬇️ Downloading {filename}...")

    response = requests.get(url, stream=True)

    with open(filepath, "wb") as f:
        for i, chunk in enumerate(response.iter_content(chunk_size=1024 * 1024)):
            if chunk:
                f.write(chunk)

                if i % 50 == 0:
                    print(f"   Downloaded {i} MB...")

    print(f"✅ Done {filename}")

if __name__ == "__main__":
    print("🚀 START DOWNLOAD...")

    for url in urls:
        download_file(url)

    print("🎉 ALL DONE!")