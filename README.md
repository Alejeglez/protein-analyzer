# protein-analyzer
Para este caso práctico, se han desarrollado 2 archivos de código. En el primero, llamado protein-analyzer.py, se ha usado el concepto de k-mers (similar a los n-gramas de PLN) para calcular todas las posibles subsecuencias de longitud k que se pueden obtener de una secuencia de aminoácidos (función generate_kmers). Antes de poder crear estas secuencias de k-mers, los datos se han leído usando la librería de BioPython, que incluye herramientas para leer secuencias de archivos fasta. De esta librería hemos sacado también las etiquetas de las proteínas usando expresiones regulares, aunque la clasificación no ha tenido un gran resultado. La función read_fasta_files() nos devuelve las secuencias, identificadores y el nombre de los archivos procesados. Los datos usados son los de las proteínas que se encontraban junto con la tarea. Se han filtrado las secuencias para quedarnos con aquellas que tengan una longitud de aminoácidos mayor a 2.

Una vez obtenemos la secuencia de k-mers para una secuencia de aminoácidos, calculamos el tf-idf de cada aminoácido en dicha secuencia. El método tf-idf (Term Frequency-Inverse Document Frequency) en el campo de PLN determina la importancia de una palabra dentro de un documento del corpus. Un valor alto de tf-idf, significa que ese fragmento es muy relevante en esa secuencia de aminoácidos, debido a que en el resto de secuencias no es tan frecuente. Para ello, usamos la librería sklearn, que ya tiene su propio método TfidfVectorizer. Esta función crea un vector para cada secuencia asignando un valor de tf-idf a cada subsecuencia. La matriz resultante de la función tfidf_method() puede ser usada para realizar clasificación. La función también muestra el fragmento más relevante para cada secuencia. 

En un estudio reciente de [Camille Moeckel et al. (2024)](https://www.sciencedirect.com/science/article/pii/S2001037024001703), se debate el uso de k-mers en los campos de la genómica y proteinómica. En él se destacan la capacidad para detectar fragmentos faltantes, mutaciones, o realizar un análisis de la presencia de estos fragmentos en genomas. También pueden ser de ayuda para detectar proteínas homólogas. Esta parte la enmarcaría en preprocesamiento de datos.

Una vez tenemos calculado el vector tf-idf para cada secuencia, podemos calcular la similitud coseno para obtener las secuencias más similares. Sklearn incluye también una función dedicada a este apartado (cosine_similarity). Si lo deseamos podemos ver los resultados en un mapa de calor, indicándolo en el parámetro plot de la función cosine_similarity_method(). Para representar el mapa de calor hemos usado las librerías Seaborn y Matplotlib. La función recibe como parámetros principales la matriz de tf-idf, así como el nombre de los archivos procesados. Los valores con mayor similitud, han sido aquellos que compartían secuencia de aminoácidos. Puede ser útil para realizar un análisis exploratorio de posibles relaciones u orígenes de proteínas. Esta parte encajaría dentro del apartado de Visualización y análisis de patrones. 

Por último, se ha intentado realizar una clasificación usando un SVM en el archivo protein-classifier.py. Sklearn incluye las funciones necesarias para ello. Para la clasificación se ha recurrido a un clasificador SVM lineal usando un 80% de los datos para entrenamiento. Los resultados no han sido muy buenos, una de las posibles explicaciones es que el conjunto de entrenamiento tiene una variabilidad de proteínas únicas grande en relación a su tamaño (24 de 38 etiquetas no se repiten, lo que implica un 63% de los datos), sumado a que para poder aplicar el k-mers se han suprimido datos. No obstante, algunos estudios (como el realizado por [Zeynep Banu Ozger (2023)](https://www.sciencedirect.com/science/article/pii/S093336572300088X)) han conseguido grandes resultados usando este enfoque. En el ejemplo citado, se consigue un accuracy de un 98.6% usando un clasificador SVM entrenado con vectores de tf-idf de 4-mers, sin embargo han usado conjuntos de datos de gran tamaño para poder llegar a este resultado. Salvo que se dispongan de estas etiquetas, sería mucho más recomendable optar por métodos de aprendizaje no supervisado.
