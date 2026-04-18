import pandas as pd


if __name__ == "__main__":
    url = 'https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet'
    df = pd.read_parquet(url)
    df.drop(columns="Unnamed: 0", inplace=True)
    df.to_parquet("data/dataset_spotify.parquet", compression='brotli')