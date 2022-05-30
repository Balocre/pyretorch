import io
import os
import pickle    

def save(obj, f: str):
    path, _ = os.path.split(f)

    os.makedirs(path, exist_ok=True)

    with io.open(f, 'wb') as opened_file:
        pickle.dump(obj, opened_file, pickle.HIGHEST_PROTOCOL)

def load(f: str):
    with io.open(f, 'rb') as opened_file:
        obj = pickle.load(opened_file)

    return obj