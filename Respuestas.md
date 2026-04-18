# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
Añade aqui tu descripción y analisis:

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset proviene de _Hugging Face_ llamado [spotify-track-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset). Este dataset contiene datos relevantes y propiedades físicas especificas de las canciones que hay dentro de Spotify. Dentro hay dos tipos de ficheros: uno en _.csv_ y otro en _.parquet_. El que se ha cogido ha sido  _0000.parquet_, el cual se ha procesado previamente por el código `parquet_processing.py` para obtener nuestro fichero en `data/dataset_spotify.parquet`. 
> La variable objetivo va a ser `popularity` debido a que es el que más sentido tiene a analizar si lo que queremos saber si, queriendo crear una nueva canción, cual podría ser su popularidad mediante sus características.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> Se presenta varios tipos de distribuciones:
> - **Distribución estándar**, las cuales presentan esta forma _'popularity'_, _'danceability'_, _'loudness'_, _'tempo'_, muy levemente _'valence'_, y a mitad en _'energy'_
> - **Distribución exponencial**, las cuales presentas esta forma _'duration_ms'_, _'acousticness'_ y _'liveness'_. Si se ignora los picos en la derecha, también serían _'speechiness'_ y _'instrumentalness'_.
> - **Distribución Bernoulli** para la de _'mode'_.
> - **Distribución multimodal** para las variables de _'key'_ y _'time_signature'_.
>
>En cuanto a los _outliers_, utilizando el método de **_z-score_** debido a que muchas de las variables no siguen una distribución estándar, se ha encontrado en las siguientes variables:
>| Variable       | _Outliers_ | Porcentaje     |
>|----------------|------------|----------------|
>| duration_ms    | 965        | 0.85%          |
>| danceability   | 157        | 0.14%          |
>| loudness       | 2465       | 2.17%          |
>| speechiness    | 2073       | 1.83%          |
>| liveness       | 3628       | 3.20%          |
>| tempo          | 201        | 0.18%          |
>| time_signature | 1130       | 1.00%          |
>
>A pesar de que sea una cantidad muy pequeña, en muchos casos tiene sentido eliminarlas, como son en el tempo o time_signature, las cuales no tiene sentido que tengan valor de 0, porque significarían que son canciones sin sonidos o sonidos audibles. Igualmente, se ha procedido eliminar todo estos _outliers_ ya que al final van a generar más ruido que ayudar al entrenamiento de la Regresión Lineal.


**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> ![correlaciones](output/ej1_heatmap_correlacion.png)
> Las tres variables con mayor correlación con la variable objetivo (_'popularity'_) son:
> - **_Instrumentalness_** con un -0.10
> - **_Loudness_** con un 0.06
> - **_Valence_** con un -0.05

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> En todo el dataset, hay solo una canción que en las columnas de 'artist', 'album_name' y 'track_name' están nulos. Su porcentaje es casi 0, y la forma de solucionarlo ha sido ver si en alguna otra fila existe el mismo 'track_id' que estuviera relleno. Tras el análisis, no se ha encontrado otra fila igual, así que se ha decidido mantener esa fila ya que son columnas que no nos dan muchos datos para el análisis.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---

Como ya se había puntualizado en el apartado anterior, nuestra variable objetivo va a ser la columna _'popularity'_, la cual indica cuán popular es una canción en Spotify. Esta variable objetivo tendrá como parámetros todas las columnas numéricas, booleanas y la columna _'track_genre'_, que indica a que genero pertenece, para el entrenamiento de una Regresión Lineal. Para ello se ha realizado lo siguiente:
1. **Carga y preprocesamiento:** Tras cargar el dataset con los outliers anteriormente eliminados, se realiza un preprocesamiento de los datos. En este preprocesamiento se ha realizado:
    - Se ha puesto todos los tipos booleanos a tipo entero (_Integer_) para que sea más fácil procesarlos.
    - Se ha transformado la columna _'track_genre'_ mediante un _LabelEncoder_. A pesar de que esta variable nominal no tenga un orden lógico, como, por ejemplo, el orden de los nombres de una membresía, se ha realizado esta transformation antes que un _OneHotEncoder_ debido a la existencia de 125 valores únicos. Esto hubiera dificultado en una cantidad considerable en el entrenamiento de la Regresión lineal.
    - Se ha eliminado las columnas que no nos resultaban interesantes de mantener para una Regresión, las cuales son _'track_id'_, _'artists'_, _'album_name'_ y _track_name'. Estas variables en su esencia son clasificadores para saber que canción estamos analizando, pero tener miles de _Label_ diferentes entorpecería más el entrenamiento que ayudarlo.
    - A continuación se ha obtenido los valores de la variable objetivo y sus parámetros (y, X) y se ha hecho un escalado estandarizado para que todas las variables estén dentro de los mismo rangos.
    - Por último se ha dividido los datos en X e y en dos partes: una parte de los datos para hacer el Entrenamiento, y la otra para hacer el Test del entrenamiento.
2. **Regresión lineal**: ya una vez obtenido los conjuntos para el Entrenamiento y el Test, se procede a entrenar nuestra regresión lineal con nuestro conjunto de Entrenamiento. Tras su entrenamiento, se ha realizado los test de MAE, RMSE y Coeficiente de determinación tanto para los conjuntos de Entrenamiento como el de Test. Junto a ello también se ha realizado para ambos conjuntos el gráfico de a nube de residuos. También se ha visto cuales son las variables más influyentes.
```
Métricas de predicción
==================================================
Entrenamiento
	MAE: 0.8252112312629936
	RMSE: 0.987137955443251
	R²: 0.02777615520546084
Test
	MAE: 0.8170351263960781
	RMSE: 0.979352008643907
	R²: 0.03202994966957806

Top 10 influencia de las variables:
             Feature  Coefficient  Absolute
11           valence    -0.110331  0.110331
9   instrumentalness    -0.103368  0.103368
5           loudness     0.067642  0.067642
2       danceability     0.064913  0.064913
3             energy    -0.062209  0.062209
7        speechiness    -0.053142  0.053142
1           explicit     0.041266  0.041266
13    time_signature     0.033455  0.033455
12             tempo     0.025690  0.025690
14       track_genre     0.019396  0.019396
```
![Nube residuos](output/ej2_residuos.png)\
Como se ve en los resultados tanto de la gráfica como de las métricas, el entrenamiento con una Regresión lineal no es la adecuada. Sin duda alguna el modelo ha sufrido un grave problema de subajuste. Como era esperable, las variables más influyentes en nuestro modelo son las mismas que se había obtenido en la gráfica de correlaciones.

En conclusión, el resultado del modelo ha sido muy malo, pero ya en el Ejercicio 1 con las correlaciones y como estaban distribuidos los datos de 'popularity' con las otras columnas, era esperable que se pudiera conseguir un buen modelo. Es probable que nuestra solución no sea usando una Regresión Lineal.\
Se podría mejorar nuestro modelo viendo cambiando de variable objetivo. Un posible candidato podría ser _'danceability'_, la cual indica lo adecuado que es una canción para ser bailado en base a su tempo, la estabilidad rítmica, la fuerza rítmica y 
su uniformidad generalizada. Por lo tanto, se podría analizar si nuevas canciones puedan usarse en, por ejemplo, discotecas, las cuales se busca canciones bailables. Esta variable, si se ve el mapa de calor de las correlaciones, tiene mejores correlaciones que _'popularity'_ por lo que nos daría un mejor modelo.

### Modelo usando _danceability_

```
Métricas de predicción
==================================================
Entrenamiento
	MAE: 0.6608590071047686
	RMSE: 0.8277739124639447
	R²: 0.3154783716549552
Test
	MAE: 0.6616785246084106
	RMSE: 0.8260929762719117
	R²: 0.3148114037157054

Top 10 influencia de las variables:
             Feature  Coefficient  Absolute
11           valence     0.491814  0.491814
3             energy    -0.349091  0.349091
8       acousticness    -0.214479  0.214479
5           loudness     0.194547  0.194547
12             tempo    -0.131065  0.131065
7        speechiness     0.125268  0.125268
10          liveness    -0.107785  0.107785
13    time_signature     0.095191  0.095191
9   instrumentalness     0.074361  0.074361
2           explicit     0.071289  0.071289
```
![Nube residuos](output/ej2_residuos_danceability.png)

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

>| Medida | Valor obtenido |
>|--------|----------------|
>| $MAE$  | $0.817035$     |
>| $RMSE$ | $0.979352$     |
>| $R^2$  | $0.032029$     |
>
> Como se ve en el `Coeficiente de determinación`, el modelo no funciona, es como si no hubiera ningún modelo. Esto es debido a, como hemos podido ver en el apartado anterior, la variable objetivo _'popularity'_ no tiene apenas correlación con alguna de las variables, y tampoco en la gráfica de puntos a pares con la variable objetivo se puede apreciar una linealidad. Por lo que nuestra solución no va a ser lineal.


---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---

![Predicciones](output/ej3_predicciones.png)\
Tras la realización de la Regresión Lineal Multiple y realizar con el conjunto de Test las predicciones, se puede ver la diferencia entre los puntos en donde se debería de posicionar y donde se han posicionado. Siendo la recta roja el caso ideal, se puede apreciar que hay muy pocos puntos en la que llegan a tocar la linea ideal. Esto podría indicar que, o la Regresión Lineal Multiple para este caso no es lo más adecuado, o que el conjunto de puntos no sea linear o que hubiera un sobreajuste significativo para el entrenamiento y dejando lugar a que nuevos valores no se puedan analizar correctamente.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> Partiendo de la formula $X\beta=y$, la parte $(X^TX)^{-1}$ proviene de la parte de $\beta$ en la que $X^TX$ se realiza para la obtención del autovector, y después su inversa para resolver la ecuación; y $X^Ty$ en la que $X^T$ viene de hacer la pseudoinversa de X, y dando lugar a la magnitud del vector. Por lo tanto, teniendo el autovector inverso y la magnitud del vector, se obtiene en donde está el intercepto y la n-pendiente de la recta.\
> En cuanto a añadir una fila de 1 a la matriz X al principio es para calcular el intercepto. Si se realizara con sólo X, se obtendría de la regresión únicamente la n-pendiente de la recta, que serían los valores de los parámetros a analizar.

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

>| Parámetro | Valor real | Valor ajustado |
>|-----------|-----------|----------------|
>| β₀        | 5.0       | 4.864995       |
>| β₁        | 2.0       | 2.063618       |
>| β₂        | -1.0      | -1.117038      |
>| β₃        | 0.5       | 0.438517       |

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

>Los valores obtenidos son los siguientes:
>
>| Medida | Valor referencia| Valor obtenido |
>|--------|-----------------|----------------|
>| $MAE$  | $1.20 (\pm 0.2)$| $1.166462$     |
>| $RMSE$ | $1.50 (\pm 0.2)$| $1.461243$     |
>| $R^2$  |$0.80 (\pm 0.05)$| $0.689672$     |
>
>Como se puede ver, tanto en el MAE como en el RMSE se aproximan al valor de >referencia, aunque no se incluya en el rango de error esperado. En cuanto al coeficiente de error es mucho menor al esperado, incluso con el rango de error. 

---

## Ejercicio 4 — Series Temporales
---

![Timeseries original](output/ej4_serie_original.png)\
En la gráfica de la serie temporal original se puede ver que hay una tendencia ascendente en el tiempo, en la que cada año aumenta los valores en cada uno de los ciclos. Se podría deducir que esta serie es adictiva, ya que no se ve que en cada periodo haya un cambio significativo en el periodo o la forma de cada ciclo.

![Timeseries descomposición](output/ej4_descomposicion.png)\
Viendo la gráfica de la descomposición se puede ver claramente lo que había extraído antes, en la que la tendencia aumenta, pero de una forma como si fuera una ecuación de cuarto grado. También se aprecia claramente los ciclos, los cuales parecen ser anuales. En cuanto al residuo da la sensación que se distribuye consistentemente en casi todo el tiempo.

Si se revisa el residuo, se podría comprobar que la descomposición nos da todos los resultados y no hay más datos que puedan estar ocultos.

![Timeseries acf pacf](output/ej4_acf_pacf.png)\
Si se realiza al residuo las evaluaciones de ACF y PACF, se puede comprobar que no hay _lag_ significativos, menos el 0. Esto indica que no hay ningún ciclo ni otras variables que con la descomposición no hayamos capturado.

![Timeseries residuo](output/ej4_histograma_ruido.png)\
Y si nos fijamos en como esta distribuido el histograma del ruido, se puede apreciar que sigue una distribución estándar. Por lo tanto el ruido sigue un patrón de variabilidad normal, y su inferencia se puede eliminar fácilmente de la serie temporal para solamente tener la tendencia y la estacionalidad.

---

### ej4_analisis.txt
```
Análisis estadístico para el residuo
==================================================
  Media  : 0.13
  Desviación Estándar : 3.22
  Asimetría   : -0.05 => Asimetría negativa
  Curtosis   : -0.06 => Curva Platicúrtica
--------------------------------------------------
  Test de normalidad Shapiro-Wilk o Jarque-Bera (p-value)  : 0.58 => Se acepta normalidad
  ADF Test para verificar estacionariedad (p-value)  : 0.00 => Se acepta estacionariedad
```
---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> Se puede apreciar que sigue una tendencia positiva, en la que en cada ciclo va aumentando los valores. Por lo tanto, es una tendencia positiva, con una dirección un poco exponencial (entre 1.1 a 1.3), siendo el incremento en su menor valor en cada ciclo de: 
> - 50
> - 60 (aumento de 10)
> - 75 (aumento de 15)
> - 95 (aumento de 20)
> - 120 (aumento de 25)
> - 140 (aumento de 20)

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> Se puede apreciar una estacionalidad, la cual se podía considerar anual, por lo tanto, con un periodo de aproximadamente 365 días. 

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> No se aprecia ciclos de largo plazo en la serie, ya que no se ve que la forma de la estacionalidad cambie abruptamente. Se diferencia de la tendencia en el sentido que la tendencia sólo guía la dirección en la que se dirige la serie, mientras que el ciclo hace que aumente o la longitud o la forma en la que se presenta cada periodo, ya sea en aumento o en decremento.  

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> Como se puede ver tanto en el análisis estadístico del residuo como en el histograma con el KDE del residuo, el ruido sería ideal. En la gráfica se ve que sigue una curva normal o estándar, y en los resultados del **p-value** ($0.58$) de Jarque-Bera nos indica que hay que rechazar el caso nulo ($H_0$), por lo tanto se acepta que una curva normal. Esto también tiene sentido en el aspecto que la **media** es cercana a 0 ($0.13$) y una **asimetría** y **curtosis** casi de 0 ($-0.05$ y $-0.06$). Por último, la **desviación estándar** es de $3.22$, lo cual solo nos indica que los datos están algo más disperso.

---

*Fin del documento de respuestas*
