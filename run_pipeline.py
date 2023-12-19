from pipelines.training_pipeline import train_pipeline

def run_training_pipeline(url: str):
    """
    Runs the training pipeline.
    
    Args:
        url (str): URL of the dataset to be used for training.
    """
    train_pipeline(url)

if __name__ == "__main__":
    dataset_url = "https://sale2.s3.us-east-2.amazonaws.com/KAG_conversion_data.csv"
    run_training_pipeline(dataset_url)
