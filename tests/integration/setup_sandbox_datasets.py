import os
import shutil
import urllib.request
import traceback

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sandbox_dir = os.path.join(base_dir, 'test_data', 'sandbox_datasets')
    
    os.makedirs(sandbox_dir, exist_ok=True)
    
    datasets = {
        'insurance.csv': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
        'titanic.csv': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    }
    
    # Try local data_splits first
    repo_root = os.path.dirname(os.path.dirname(base_dir))
    local_sources = {
        'insurance.csv': os.path.join(repo_root, 'data_splits', 'insurance_train.csv'),
        'titanic.csv': os.path.join(repo_root, 'data_splits', 'Titanic_train.csv')
    }
    
    for filename, url in datasets.items():
        dest_path = os.path.join(sandbox_dir, filename)
        if os.path.exists(dest_path):
            print(f"Skipping {filename}, already exists")
            continue
            
        local_src = local_sources.get(filename)
        if local_src and os.path.exists(local_src):
            print(f"Copying {filename} from {local_src}")
            shutil.copy2(local_src, dest_path)
            continue
            
        print(f"Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
