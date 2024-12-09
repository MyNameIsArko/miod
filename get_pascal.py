import os.path
import tarfile

import requests
from tqdm.auto import tqdm
import shutil

location = "pascal"

def download(url, location):
    local_filename = url.split('/')[-1]

    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(f"{location}/{local_filename}", 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            chunk_size = 500_000_000  # 0.5 GB
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size), desc=f"Downloading {local_filename}",
                              total=(total_length // chunk_size) + 1):
                f.write(chunk)


def extract(location, file):
    print(f"Extracting {file}")
    with tarfile.open(f"{location}/{file}", "r:*") as tar_ref:
        tar_ref.extractall(location, filter="data")

    print(f"Moving folders to {location}")
    source_dir = f'{location}/VOCdevkit/VOC2012'
    target_dir = f'{location}'

    file_names = os.listdir(source_dir)

    for file_name in tqdm(file_names):
        shutil.move(os.path.join(source_dir, file_name), target_dir)

def remove(location, file):
    shutil.rmtree(f"{location}/VOCdevkit")
    print("Old folder removed")

    os.remove(f"{location}/{file}")
    print(f"{file} removed")

def get_pascal():
    os.makedirs(location, exist_ok=True)

    if not os.path.isdir(f"{location}"):
        file = "VOCtrainval_11-May-2012.tar"
        url = f"http://host.robots.ox.ac.uk/pascal/VOC/voc2012/{file}"
        download(url, location)
        extract(location, file)
        remove(location, file)
    else:
        print(f"Pascal VOC already exists")

if __name__ == '__main__':
    get_pascal()