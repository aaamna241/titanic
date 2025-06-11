"""

"""

from src.titanic.data import load_data, clean_data, prepare_data
from src.titanic.registry import save_model
from src.titanic.train import train_model, evaluate_model, optimize_model
# for train dataset
df = load_data("train.csv")
clean_df = clean_data(df)
train = prepare_data(clean_df)
# for test dataset
df_test = load_data("test.csv", is_test=True)
clean_df_test = clean_data(df_test)
test = prepare_data(clean_df_test)

print(train)
print(test)

# Train the model
model= train_model(train)
score = evaluate_model(model, test)
print(f"Model score: {score}")
