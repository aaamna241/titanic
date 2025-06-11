"""

"""

from src.titanic.data import load_data, clean_data, prepare_data
from src.titanic.registry import save_model, load_model
from src.titanic.train import train_model, evaluate_model, optimize_model

def process_dataset(file_path, is_test=False):
    df = load_data(file_path, is_test=is_test)
    clean_df = clean_data(df)
    return prepare_data(clean_df)

def main():
    train = process_dataset("train.csv")
    test = process_dataset("test.csv", is_test=True)

    # Train and evaluate model
    model = train_model(train)
    print(f"Model score: {evaluate_model(model, test)}")

    # Optimize and re-evaluate
    optimized_model = optimize_model(model, train)
    print(f"Optimized model score: {evaluate_model(optimized_model, test)}")

    # Save and reload model
    save_model(optimized_model, "models")
    load_model("models")

if __name__ == "__main__":
    main()
