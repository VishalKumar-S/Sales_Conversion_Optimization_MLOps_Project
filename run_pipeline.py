from pipelines.training_pipeline import train_pipeline


if __name__ == "__main__":
    train_pipeline("https://sale2.s3.us-east-2.amazonaws.com/KAG_conversion_data.csv")
