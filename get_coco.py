import os.path

import requests
from tqdm.auto import tqdm
from zipfile import ZipFile

location = "coco"

def download(url, location):

    local_filename = url.split('/')[-1]

    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(f"{location}/{local_filename}", 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            chunk_size = 2_000_000_000  # 2 GB
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size), desc=f"Downloading {local_filename}",
                              total=(total_length // chunk_size) + 1):
                f.write(chunk)


def extract(location, file):
    print(f"Extracting {file}")
    with ZipFile(f"{location}/{file}") as zip_ref:
        zip_ref.extractall(location)

def remove(location, file):
    os.remove(f"{location}/{file}")
    print(f"{file} removed")

def get_coco(subset, year):
    os.makedirs(location, exist_ok=True)

    if not os.path.isdir(f"{location}/{subset}{year}"):
        file = f"{subset}{year}.zip"
        url = f"http://images.cocodataset.org/zips/{file}"
        download(url, location)
        extract(location, file)
        remove(location, file)
    else:
        print(f"{subset}{year} already exists")

    if subset in ['train', 'val']:
        if not os.path.isdir(f"{location}/annotations"):
            file = f"annotations_trainval{year}.zip"
            url = f"http://images.cocodataset.org/annotations/{file}"
            download(url, location)
            extract(location, file)
            remove(location, file)
        else:
            print("annotations already exists")

if __name__ == '__main__':
    get_coco("train", 2017)
    get_coco("val", 2017)
    get_coco("test", 2017)