import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def main(args):
    # Crear directorio de salida del modelo si no existe
    os.makedirs(args.output_model_path, exist_ok=True) [cite: 26]

    input_train_file = os.path.join(args.input_data_path, 'train_data.csv')
    # input_test_file = os.path.join(args.input_data_path, 'test_data.csv') # Opcional si se evalúa aquí

    print(f"Leyendo datos de entrenamiento desde: {input_train_file}")
    train_df = pd.read_csv(input_train_file)

    X_train = train_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train_df['species_encoded']

    print("Entrenando el modelo de Regresión Logística...")
    model = LogisticRegression(random_state=args.random_seed, max_iter=args.max_iter, solver='liblinear')
    model.fit(X_train, y_train)

    # Evaluación (opcional aquí, podría ser un paso separado del pipeline) [cite: 27]
    # test_df = pd.read_csv(input_test_file)
    # X_test = test_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    # y_test = test_df['species_encoded']
    # y_pred = model.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print(f"Accuracy en datos de prueba: {acc:.4f}")
    # print(classification_report(y_test, y_pred))

    model_filename = os.path.join(args.output_model_path, 'model.joblib')
    print(f"Guardando el modelo entrenado en: {model_filename}")
    joblib.dump(model, model_filename)

    print("Entrenamiento completado.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Iris Model')
    parser.add_argument('--input-data-path', type=str, required=True, help='Path to the processed data folder (containing train_data.csv).')
    parser.add_argument('--output-model-path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for model training.')
    parser.add_argument('--max-iter', type=int, default=200, help='Maximum iterations for Logistic Regression.')

    args = parser.parse_args()
    main(args)