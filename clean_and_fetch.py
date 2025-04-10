import os
import shutil
import subprocess
from src.config import DATA_DIR, MODELS_DIR, RESULTS_DIR

def clean_directories():
    """Clean all data, models, and results directories"""
    dirs_to_clean = [DATA_DIR, MODELS_DIR, RESULTS_DIR]
    
    for directory in dirs_to_clean:
        if os.path.exists(directory):
            print(f"Cleaning directory: {directory}")
            # Remove all files in the directory
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        print(f"  Deleted: {file}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"  Deleted directory: {file}")
                except Exception as e:
                    print(f"  Error deleting {file_path}: {e}")
        else:
            print(f"Creating directory: {directory}")
            os.makedirs(directory)
    
    print("All directories cleaned successfully!")

if __name__ == "__main__":
    # Clean all data, models, and results
    clean_directories()
    
    # Fetch fresh market data
    print("\nFetching fresh market data...")
    subprocess.run(["python", "fetch_3years_data.py"], check=True)
    
    # Run the model with the fresh data
    print("\nRunning model with fresh data...")
    import improved_hmm_model_v2
    improved_hmm_model_v2.run_improved_model_v2(crypto='BTC', interval='4h')
    
    print("Process complete: All data deleted and fresh data fetched and processed.") 