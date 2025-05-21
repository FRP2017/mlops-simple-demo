import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main(args):
    # Crear directorios de salida si no existen
    os.makedirs(args.output_data_path, exist_ok=True)

    print(f"Leyendo datos desde: {args.input_data_path}")
    df = pd.read_csv(args.input_data_path)

    print("Realizando división train-test...")
    # Mapear especies a números para el modelo
    species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    df['species_encoded'] = df['species'].map(species_map)

    # Eliminar filas con NaN en 'species_encoded' si alguna especie no estaba en el map
    df.dropna(subset=['species_encoded'], inplace=True)
    df['species_encoded'] = df['species_encoded'].astype(int)


    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_split_ratio, random_state=args.random_seed)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    output_train_path = os.path.join(args.output_data_path, 'train_data.csv')
    output_test_path = os.path.join(args.output_data_path, 'test_data.csv')

    print(f"Guardando datos de entrenamiento en: {output_train_path}")
    train_df.to_csv(output_train_path, index=False)

    print(f"Guardando datos de prueba en: {output_test_path}")
    test_df.to_csv(output_test_path, index=False)

    print("Preprocesamiento completado.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Iris Data')
    parser.add_argument('--input-data-path', type=str, required=True, help='Path to the raw input CSV file (GCS URI or local).')
    parser.add_argument('--output-data-path', type=str, required=True, help='Path to save the processed data (GCS URI or local folder).')
    parser.add_argument('--test-split-ratio', type=float, default=0.2, help='Ratio for the test split.')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for train-test split.')

    args = parser.parse_args()
    main(args)