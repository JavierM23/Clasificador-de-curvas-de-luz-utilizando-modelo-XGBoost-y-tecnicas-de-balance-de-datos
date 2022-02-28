# Clasificador de curvas de luz utilizando modelo XGBoost y tecnicas de balance de datos

Los telescopios recolectan todas las noches grandes cantidades de datos sobre las variaciones en el brillo de objetos estelares o bien, de su movimiento, denominadas alertas astronómicas. Dado el volumen de datos y la velocidad con la que se produce se requieren agentes intermediarios, denominados *brokers*, que son quienes realizan la clasificación de alertas.
    
ALeRCE es un *broker* que recibe las alertas provenientes del \textit{survey} astronómico ZTF (*Zwicky Transient Facility*) y entre sus principales labores esta la rápida clasificación de las alertas, siendo capaz de separar las alertas falsas de las reales, y dentro de estas últimas, identificar hasta 15 clases distintas.
    
En el presente trabajo se evaluó el potencial del algoritmo de aprendizaje XGBoost como modelo predictivo como clasificador de curvas de luz. Actualmente ALeRCE utiliza el modelo *Balanced Random Forrest* (BRF). La motivación detrás de este estudio es el gran desbalance de los datos, el cual es agravado debido a las múltiples clases existentes. Por este motivo se propone el estudio e implementación de técnicas para evitar el efecto de entrenar modelos con desbalance de datos
    
Para el entrenamiento del modelo XGBoost usando distintas técnicas de balance, se implementó el procedimiento *Nested Cross Validation* mediante el cual se entrena y evalúa cada modelo 10 veces con distintos grupos de entrenamiento y test, de forma de obtener valores promedio para el desempeño. Este mismo procedimiento fue realizado además para replicar el clasificador de ALeRCE con BRF, para así poder comparar el desempeño de ambos modelos.
    
Combinando varias técnicas de desbalance con XGBoost se obtuvieron muy buenos resultados en las métricas de evaluación y en sus predicciones. Al analizar las matrices de confusión resultantes se comprobó una disminución en el sesgo hacia las clases mayoritarias por parte del modelo predictivo. Se destaca la técnica a nivel de algoritmo *Cost Sensitive Learning*, con la cual XGBoost superó a BRF en todos clasificadores que componen el clasificador de curvas de luz, obteniendo en el nivel inferior del clasificador valores de 0.67, 0.79 y 0.70 para Precision, Recall y F1-score respectivamente, en contraste con BRF que obtuvo valores de 0.57, 0.76 y 0.60 para las mismas métricas respectivamente, demostrando además que las diferencias de desempeño obtenidas fueron estadísticamente significativas.
    
Por último, se realizó un test final en los modelos entrenados con los que se obtuvo mejores resultados. Las curvas de luz de este test corresponden a aquellas que no fueron utilizadas ni durante la experimentación del presente trabajo ni por ALeRCE en el paper en el que presentaron su clasificador de luz.

Javier Molina F.
javier.molina.ferreiro@gmail.com
