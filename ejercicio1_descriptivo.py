import pandas as pd
import numpy as np
import math
from scipy.stats import iqr, skew, kurtosis, zscore
import matplotlib.pyplot as plt
import seaborn as sns

SEMILLA = 42

def __get_rows_cols_memory(df:pd.DataFrame):
    """
    Método privado para la obtención de la forma (filas y columnas) de un dataframe junto a su uso en memoria RAM.

    Parámetros
    ----------
    dataframe: pd.DataFrame —  El dataframe el cual queremos obtener las características

    Retorna
    -------
    shape: (int, int) — La forma que tiene el dataframe
    memory: int — La memoria RAM usada en Bytes.

    """
    shape, memory = df.shape, df.memory_usage(index=False).sum() * (10**-6)
    print("Número de filas y columnas:", shape)
    print(f"Tamaño en memoria: {memory:.5f} MB")
    return shape, memory


def resumen_estructural(dataframe: pd.DataFrame):
    """
    Obtención y representación de las características de nuestro dataframe.
    Junto a ello se realiza ciertas operaciones, como la eliminación de duplicados
    o eliminación de valores nulos, cuando son necesarios.

    Parámetros
    ----------
    dataframe: pd.DataFrame — El dataframe el cual queremos obtener las características

    """
    print("Resumen estructural")
    print("="*50)
    shape, _ = __get_rows_cols_memory(dataframe)
    print()
    print("Tipos de datos:")
    print(dataframe.dtypes)
    print()
    print(f"Cantidad de duplicados: {dataframe.duplicated().sum()}")
    print(f"Porcentaje de duplicados: {100 * dataframe.duplicated().sum()/shape[0]:.4f}%")
    dataframe.drop_duplicates(keep="first",inplace=True)
    print()
    filas, columnas = dataframe.shape
    print("Tras eliminar los duplicados")
    shape, _ = __get_rows_cols_memory(dataframe)
    print()
    print("Cantidad de nulos:")
    print(dataframe.isna().sum())
    print()
    # Se ve que solo hay un valor nulo para los campos 'artists', 'album_name' y 'track_name'
    # Vamos a ver cuales son las canciones con esos valores nulo
    set_trackid = set()
    for col in ['artists','album_name','track_name']:
        #Obtenemos el track_id de la fila que tuviera ese valor nulo
        set_trackid.add(dataframe.loc[dataframe.isna()[col],'track_id'].values[0])
    print("Track_id con algún valor nulo:", set_trackid)
    #Se ve solo hay una canción que tiene esos campos a nulo. Por lo que se va a ver si otra fila contiene ese mismo track_id
    print("Filas con el track_id anterior")
    print(dataframe[dataframe['track_id'].isin(set_trackid)])
    #Se ve que solo existe una fila con ese track_id, por lo tanto no hay forma de rellenar ese valor nulo
    #Como es en columnas de variables cualitativa nominal, y es solamente 1, no se va a eliminar la fila

def __type_asymmetry(asymmetry: float) -> str:
    """
    Método privado para la obtención del tipo de simetría a través de su coeficiente.

    Parámetros
    ----------
    asymmetry: float —  El coeficiente de asimetría

    Retorna
    -------
    str — La cadena de texto que indica que tipo de asimetría es

    """
    if asymmetry == 0: return "Simetría"
    if asymmetry > 0: return "Asimetría positiva"
    return "Asimetría negativa"

def __type_kurtosis(kurtosis: float) -> str:
    """
    Método privado para la obtención del tipo de curtosis a través de su coeficiente.

    Parámetros
    ----------
    kurtosis: float —  El coeficiente de curtosis

    Retorna
    -------
    str — La cadena de texto que indica que tipo de curtosis es

    """

    if kurtosis == 0: return "Curva Mesocúrtica"
    if kurtosis > 0: return "Curva Leptocúrtica"
    return "Curva Platicúrtica"

def __statistics(dataframe:pd.DataFrame, var_objetivo:str, file: str = 'output/ej1_descriptivo.csv'):
    """
    Método privado para la obtención de las características estadísticas: la media, la mediana, la desviación típica, Q1, Q3, minimizo
    y máximo, el IQR de la variable objetivo y sus coeficientes de asimetría y curtosis.

    Parámetros
    ----------
    dataframe: pd.DataFrame — El dataframe el cual queremos obtener las características
    var_objetivo: str — El String de la variable objetivo a analizar
    file: str — La ruta en la que se va a guardar en formato csv los valores estadísticos.

    Retorna
    -------
    list — Lista con las variables numéricas identificadas al hacer el describe()

    """
    df_st = dataframe.describe()
    df_st.loc['median'] = np.median(dataframe[df_st.columns].values, axis=0)
    df_st.loc['variance'] = np.var(dataframe[df_st.columns].values, axis=0)
    df_st.to_csv(file)
    print("Tabla estadística descriptiva:")
    print(df_st.T)
    print()
    print(f"Rango intercuartílico de la variable objetivo ({var_objetivo}):", iqr(dataframe[var_objetivo]))
    asimetria = skew(dataframe[var_objetivo])
    print(f"Coeficiente de asimetría en {var_objetivo}: {asimetria:.5f}\nTipo de asimetría: {__type_asymmetry(asimetria)}")
    curtosis = kurtosis(dataframe[var_objetivo])
    print(f"Curtosis en {var_objetivo}: {curtosis:.5f}\nTipo de curtosis: {__type_kurtosis(curtosis)}")
    print()
    return df_st.columns


def __histogram_plot(dataframe:pd.DataFrame, file: str, cols:list = None, n_cols: int = 5, figsize:tuple = None, log_scale:bool = False, n_bins:(int|str|list)="auto"):
    """
    Método privado para la representación de los histogramas de las variables numéricas.

    Parámetros
    ----------
    dataframe: pd.DataFrame — El dataframe en donde se obtiene los datos
    file: str — La ruta en la que se va a guardar en formato csv los valores estadísticos.
    n_cols: int — El número de columnas que se van a crear para el subplot
    figsize: (int,int) — Tupla que indica el tamaño de las gráficas del subplot
    log_scale: bool — Booleano que indica si se va a realizar una escala logarítmica en el conteo
    n_bins: (int|str|list) — El numero de barras que se va a representar en la gráfica.

    """
    
    if cols is None: cols = dataframe.columns
    n_rows = math.ceil(len(cols)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    flat_axes = np.ravel(axes)

    for ax, col in zip(flat_axes, cols):
        # Se comprueba si la columna esta formados por integer con pocas variables, ya que no tiene sentido 
        # hacer un histograma continuo con columnas de pocas variables discretas
        if dataframe[col].dtype == np.int64 and dataframe[col].nunique() < n_bins:
            sns.histplot(dataframe[col], ax=ax, discrete=True)
        else:
            sns.histplot(dataframe[col], ax=ax, bins=n_bins)
        # Si hay valores muy irregulares, se va a hacer log del conteo
        if log_scale: 
             ax.set_yscale('log')
             ax.set_ylabel('Log Count')
        ax.set_title(f"{col.replace('_',' ').capitalize()}")
        ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(file, dpi=300, bbox_inches='tight')

def __boxplot(dataframe:pd.DataFrame, file: str, cols:list = None, n_cols: int = 5, figsize:tuple = None):
    """
    Método privado para la representación de los boxplots

    Parámetros
    ----------
    dataframe: pd.DataFrame — El dataframe en donde se obtiene los datos
    file: str — La ruta en la que se va a guardar en formato csv los valores estadísticos.
    cols: list(str) — Lista de los nombres de las columnas a representar
    n_cols: int — El número de columnas que se van a crear para el subplot
    figsize: (int,int) — Tupla que indica el tamaño de las gráficas del subplot

    """

    if cols is None: cols = dataframe.columns
    n_rows = math.ceil(len(cols)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    flat_axes = np.ravel(axes)

    for ax, col in zip(flat_axes, cols):
        sns.boxplot(dataframe[col], ax=ax)
        ax.set_title(f"{col.replace('_',' ').capitalize()}")
        ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(file, dpi=300, bbox_inches='tight')

def __outliers(dataframe: pd.DataFrame, cols:list = None):
    """
    Método privado para la obtención de los outliers de un dataframe

    Parámetros
    ----------
    dataframe: pd.DataFrame — El dataframe en donde se obtiene los datos
    cols: list(str) — Lista de los nombres de las columnas a representar

    Retorna
    -------
    pd.DataFrame — Dataframe con los outliers eliminados

    """
    print("Outliers")
    if cols is None: cols = dataframe.columns
    set_rows = set()
    for col in cols:
        z_score = zscore(dataframe[col])
        outliers = dataframe[np.abs(z_score)>3]
        print(f"Outliers en {col}: {len(outliers)} ({100*len(outliers)/len(dataframe[col]):.2f}%)")
        set_rows.update(list(outliers.index))
    return dataframe.drop(list(set_rows))
    


def descripcion_estadistica(dataframe: pd.DataFrame, var_objetivo:str):
    """
    Obtención de los datos estadísticos de un Dataframe, con observaciones en la variable objetivo

    Parámetros
    ----------
    dataframe: pd.DataFrame — El dataframe el cual queremos obtener las características
    var_objetivo: str — Nombre de la variable objetivo

    Retorna
    -------
    dataframe: pd.DataFrame — Dataframe con los outliers eliminados
    numeric_cols: list(str) — Lista con los nombres de las columnas numéricas

    """
    print("Estadísticos descriptivos de variables numéricas")
    print("="*50)
    numeric_cols = __statistics(dataframe, var_objetivo)
    __histogram_plot(dataframe,'output/ej1_histogramas.png', numeric_cols, log_scale=True, figsize=(20,10), n_bins=30)
    __boxplot(dataframe,'output/ej1_boxplots.png', numeric_cols, figsize=(10,20))
    dataframe = __outliers(dataframe,numeric_cols)
    return dataframe, numeric_cols

def variables_categoricas(dataframe: pd.DataFrame):
    """
    Obtención de los datos estadísticos de un Dataframe para las variables categóricas.

    Parámetros
    ----------
    dataframe: pd.DataFrame — El dataframe el cual queremos obtener las características

    """
    print("Análisis variables categóricas")
    df_categorical = dataframe.select_dtypes(include=['string', 'boolean'])
    print("Frecuencia absoluta y relativa")
    for col in df_categorical.columns:
        print(df_categorical[col].value_counts())
        print(df_categorical[col].value_counts(normalize=True))

        plt.figure(figsize=(20,5))
        df_categorical[col].value_counts()[:150].plot(kind="bar")
        plt.xlabel(f"{col.replace('_',' ').capitalize()}")
        plt.xticks(rotation=50, ha='right', fontsize=7)
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig(f'output/ej1_categoricas_{col}.png', dpi=150, bbox_inches='tight')

def correlations(dataframe: pd.DataFrame, var_objetivo:str):
    """
    Obtención de las correlaciones de un Dataset y sus relaciones con la variable objetivo

    Parámetros
    ----------
    dataframe: pd.DataFrame — El dataframe el cual queremos obtener las características
    var_objetivo: str — Nombre de la variable objetivo

    """
    corr = dataframe.corr(numeric_only=True)
    plt.figure(figsize=(10,10))
    ax = sns.heatmap(
        corr,
        annot=True, fmt=".2f",
        cmap="vlag", center=0,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": .8, "label": "Correlación"}
    )
    ax.set_title("Mapa de calor de Correlación")
    plt.tight_layout()
    plt.savefig('output/ej1_heatmap_correlacion.png', dpi=300, bbox_inches='tight')
    
    cols_numerical = corr.columns
    fig, axes = plt.subplots(5, 3, figsize=(15,30), layout='constrained', sharey=True)

    flat_axes = np.ravel(axes)
    for ax, col in zip(flat_axes, cols_numerical):
        sns.scatterplot(dataframe, x=col, y=var_objetivo, ax=ax)
        ax.set_title(f"{col.replace('_',' ').capitalize()}")
    fig.suptitle('Comparación entre variable objetivo y las demás', fontsize=16)
    plt.savefig('output/ej1_objetive_vs_rest.png', dpi=300, bbox_inches='tight')

def main():
    df = pd.read_parquet("data/dataset_spotify.parquet") 
    resumen_estructural(df)
    print('-'*50)
    df, _ = descripcion_estadistica(df,var_objetivo='popularity')
    df.to_parquet("data/dataset_spotify_wo_outliers.parquet", compression='brotli')
    print('-'*50)
    variables_categoricas(df)
    print('-'*50)
    correlations(df, var_objetivo='popularity')



if __name__ == '__main__':
    main()