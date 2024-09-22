from sklearn.preprocessing import LabelEncoder


def preprocesar_datos(train_data, test_data):
    label_encoder = LabelEncoder()

    # Convertir todas las columnas categóricas a numéricas
    for column in train_data.columns:
        train_data[column] = label_encoder.fit_transform(train_data[column])
        test_data[column] = label_encoder.transform(test_data[column])

    # Asegurarse de que los hongos de comestibilidad desconocida sean asignados a la clase de hongos venenosos
    train_data["type"] = train_data["type"].apply(lambda x: 1 if x == "p" else 0)
    test_data["type"] = test_data["type"].apply(lambda x: 1 if x == "p" else 0)

    return train_data, test_data
