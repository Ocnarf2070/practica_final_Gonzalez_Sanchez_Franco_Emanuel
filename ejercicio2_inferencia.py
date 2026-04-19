import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


SEMILLA = 42


def __preprocessing(df:pd.DataFrame, var_objetivo: str) -> tuple:
    """
    Método privado para el preprocesamiento de los datos. Este preprocesamiento consiste en el cambio de 
    tipos adecuados, realizar Encoders de valores categóricos, eliminación de columnas no necesarias,
    estandarización de las variables y separación del conjunto en dos (Train y Test).

    Parámetros
    ----------
    df: pd.DataFrame — El dataframe en el que se va a aplicar las trasformaciones
    var_objetivo: str — Nombre de la variable objetivo

    Retorna
    ---------
    df: pd.DataFrame — El dataframe con las transformaciones
    X_train, X_test, y_train, y_test: np.array — Los conjuntos estandarizados para los parámetros y variable objetivo dividido entre Entrenamiento y Test

    """

    # Se ve que hay un tipo booleano, que es valido para la columna 'explicit', pero se va a cambiar a int para poder tratar mejor  
    # este dato en siguientes secciones
    df['explicit'] = df['explicit'].astype(int)

    # A pesar de ser un tipo nominal sin un orden claro, se le va a ser un Encoder
    # Debido a la cantidad de valores únicos presentes (aprox. 150 valores)
    df['track_genre'] = LabelEncoder().fit_transform(df['track_genre'])
    df.drop(columns=["track_id", "artists", "album_name", "track_name"], inplace=True)
    
    # Se sacan los valores de la variable objetivo y las variables a usar en la regresión
    y = df[var_objetivo].values
    X = df.drop(columns=var_objetivo).values

    #Se realiza un escalado estandarizado para tener una consistencia entre todas las variables
    X_standard = StandardScaler()
    y_standard = StandardScaler()
    X_std = X_standard.fit_transform(X)
    y_std = y_standard.fit_transform(y[:, np.newaxis]).flatten()

    #Se divide tanto la X como la y para una parte de entrenamiento y otra para el test
    X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.2, random_state=SEMILLA)
  
    return df, X_train, X_test, y_train, y_test 
    

def __residuos(y_pred_train: np.array, y_pred_test: np.array, y_train: np.array, y_test:np.array):
    """
    Método privado para la representación del residuo entre la predicción y el valor real
    tanto para el conjunto de Entrenamiento como en el de Test.

    Parámetros
    ----------
    y_pred_train, y_pred_test: np.array — Array que contiene las predicciones del conjunto correspondiente.
    y_train, y_test: np.array — Array que contiene los valores reales del conjunto correspondiente

    """
    
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
    """
    Método privado para la obtención de las métricas MAE, RMSE y Coeficiente de determinación tanto para
    los conjuntos de Entrenamiento como de Test. También realiza el análisis y representación del residuo
    de las predicciones y los valores reales.

    Parámetros
    ----------
    df: pd.DataFrame — El dataframe en el que se va a realizar el análisis
    modelo: LinearRegression — El modelo de la Regresión Lineal entrenada
    X_train, y_train: np.array — Arrays perteneciente al conjunto de Entrenamiento que contiene los valores de los parámetros y los valores de la variable objetivo
    X_test, y_test: np.array — Arrays perteneciente al conjunto de Test que contiene los valores de los parámetros y los valores de la variable objetivo

    """
    
    #Calculamos como de bien ha ido la regresión usando el conjunto de el test como el entretenimiento
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
    
    __residuos(y_pred_train, y_pred_test, y_train, y_test)

def linear_regression(df:pd.DataFrame, var_objetivo:str):
    """
    Función que realiza la Regresión linear del dataframe pasado y su variable objetivo. 
    Para ello realiza primero el preprocesamiento de los datos, después el entrenamiento 
    de la Regresión Lineal, y por último el análisis con sus métricas del modelo obtenido.
    
    Parámetros
    ----------
    dataframe: pd.DataFrame — El dataframe el cual queremos realizar la Regresión Lineal
    var_objetivo: str — Nombre de la variable objetivo

    """

    df, X_train, X_test, y_train, y_test = __preprocessing(df, var_objetivo)
    np.random.seed(SEMILLA)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Revisamos cuales son los features que han salido 
    coef_df = pd.DataFrame({
        'Feature': df.drop(columns=var_objetivo).columns,
        'Coefficient': lr.coef_,
        'Absolute': np.abs(lr.coef_)
    })
    # Y los ordenamos cuales son los más importantes, independientemente que sean
    # positivos o negativos
    print(coef_df.sort_values(by='Absolute', ascending=False)[:10])

    __metricas(df, lr, X_train, y_train, X_test, y_test)
    
def init_dataset(file: str) -> pd.DataFrame:
    """
    Carga el dataset indicado, realizando la carga correcta si es tipo csv o parquet.

    Parámetros
    ----------
    file: str — Texto que indica la ubicación del fichero a cargar

    Retorna
    ---------
    df: pd.DataFrame — El dataframe del fichero cargado con los datos.
    """
    if '.csv' in file:
        return pd.read_csv(file)
    elif '.parquet' in file:
        return pd.read_parquet(file)
    else:
        raise Exception("Format not implemented")
    
def main():
    df = init_dataset("data/dataset_spotify_wo_outliers.parquet")
    linear_regression(df, 'popularity')


if __name__ == "__main__":
    main()
    