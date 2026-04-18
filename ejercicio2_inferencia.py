import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


SEMILLA = 42


def init_dataset(file: str) -> pd.DataFrame:
    df = pd.read_parquet(file)
    # Se ve que hay un tipo booleano, que es valido para la columna 'explicit', pero se va a cambiar a int para poder tratar mejor  
    # este dato en siguientes secciones
    df['explicit'] = df['explicit'].astype(int)

    # A pesar de ser un tipo nominal sin un orden claro, se le va a ser un Encoder
    # Debido a la cantidad de valores únicos presentes (aprox. 150 valores)
    df['track_genre'] = LabelEncoder().fit_transform(df['track_genre'])
    df.drop(columns=["track_id", "artists", "album_name", "track_name"], inplace=True)
    return df

def __residuos(y_pred_train, y_pred_test,y_train,y_test):
    residual_train = y_train - y_pred_train
    residual_test = y_test - y_pred_test
    plt.figure(figsize=(10,5))
    plt.scatter(y_pred_train, residual_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Train')
    plt.scatter(y_pred_test, residual_test,
                c='limegreen', marker='s', edgecolor='white',
                label='Test', alpha=0.6)
    plt.xlabel('Predicción popularidad')
    plt.ylabel('Residual')
    plt.legend(loc='best')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Nube de puntos residuales')
    plt.savefig("output/ej2_residuos.png", dpi=300, bbox_inches='tight')

def __metricas(df:pd.DataFrame, modelo: LinearRegression, X_train:np.array, y_train:np.array, X_test: np.array, y_test: np.array):
    y_pred_train = modelo.predict(X_train)
    MAE_train = mean_absolute_error(y_train, y_pred_train)
    RMSE_train = root_mean_squared_error(y_train, y_pred_train)
    R2_train = r2_score(y_train, y_pred_train)
    y_pred_test = modelo.predict(X_test)
    MAE_test = mean_absolute_error(y_test, y_pred_test)
    RMSE_test = root_mean_squared_error(y_test, y_pred_test)
    R2_test = r2_score(y_test, y_pred_test)
    with open("output/ej2_metricas_regresion.txt", "w", encoding="utf-8") as f:
        f.write("Métricas de predicción\n")
        f.write('='*50)
        f.write('\n')
        f.write('Entrenamiento\n')
        f.write(f"\tMAE: {MAE_train}\n")
        f.write(f"\tRMSE: {RMSE_train}\n")
        f.write(f"\tR²: {R2_train}\n")
        f.write('Test\n')
        f.write(f"\tMAE: {MAE_test}\n")
        f.write(f"\tRMSE: {RMSE_test}\n")
        f.write(f"\tR²: {R2_test}\n")

    coef_df = pd.DataFrame({
        'Feature': df.drop(columns='popularity').columns,
        'Coefficient': modelo.coef_
    })
    print(coef_df.sort_values(by='Coefficient', ascending=False)[:10])
    __residuos(y_pred_train, y_pred_test, y_train, y_test)


    

def linear_regression(df:pd.DataFrame, var_objetivo:str):
    y = df[var_objetivo].values
    X = df.drop(columns=var_objetivo).values
    X_standard = StandardScaler()
    y_standard = StandardScaler()
    X_std = X_standard.fit_transform(X)
    y_std = y_standard.fit_transform(y[:, np.newaxis]).flatten()
    X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.2, random_state=SEMILLA)
  
    np.random.seed(SEMILLA)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    __metricas(df, lr, X_train, y_train, X_test, y_test)
    


def main():
    df = init_dataset("data/dataset_spotify_wo_outliers.parquet")
    linear_regression(df, 'popularity')



if __name__ == "__main__":
    main()
    