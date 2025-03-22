from pipelines.training_pipeline import train_pipeline

def run_training_pipeline(path: str):
    """
    Runs the training pipeline.
    """
    train_pipeline(path)

if __name__ == "__main__":
    dataset_path = "data/KAG_conversion_data.csv"    
    run_training_pipeline(dataset_path)
