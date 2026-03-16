# Proyecto 6: Voz en la Nube - Entrenamiento de voces sintéticas

**Por:** Luis Santiago Brenes Ruiz

---

## Entorno de trabajo

Una de las partes más importantes al entrenar o generar un modelo TTS, ya que estos requieren altas capacidades de cómputo generalmente proporcionadas por GPUs, lo cual se puede hallar en algunas aplicaciones web tanto gratis como de pago.

Para este proyecto inicialmente se utilizó Google Colab con un plan gratuito, pero tras algunas pruebas con varios de los modelos se notó que las capacidades no eran suficientes, la memoria se borra al cerrar la sesión y además se desconocen los límites diarios inclusive pagando un plan profesional.

Tras una búsqueda se optó por Kaggle, igualmente con un plan gratuito, pero que ofrece 30 horas semanales de uso en GPUs de distinto tipo (2xT4 y P100) y persistencia en el almacenamiento (20GB).

---

## Modelos probados

De los siguientes modelos mencionados, en ninguno fue posible obtener resultados consistentes, ya sea por dificultades en la instalación, fallas en los entrenamientos, en la generación de voz o resultados incoherentes.

Cabe mencionar que todos fueron probados con el mismo dataset de voz seleccionado aleatoriamente del repositorio del TCU: [2024-SSV-002](https://6f33fa7f78ea46e2aaca.sharepoint.com/:f:/s/TCU-748/IgDV5L2k2jplSIVw6qpaaSx1Ab7Cyao3Zh0hUZkrJj5ywik?e=g7l3Q9). Todos en formato `.wav` con frecuencia de muestreo de **32kHz**, y transcripción en `.txt` con encoding **windows-1252**.

---

### SopranoTTS

Repositorio:

- https://github.com/ekwek1/soprano  
- https://github.com/ekwek1/soprano-factory

Se seleccionó ya que involucra 2 repositorios, uno con el modelo y el otro con las instrucciones para el entrenamiento personalizado (Soprano-Factory), además de tener actualizaciones recientes.

El ejemplo que se incluye (voz en inglés) funciona correctamente en Google Colab y es posible ejecutar el entrenamiento en el mismo entorno, sin embargo, parece que no está del todo actualizado o documentado ya que no genera un archivo esencial (`decoder.pth`).

Según comentarios en internet, no es lógico usar el archivo original ya que no está entrenado para español u alguna voz específica, además otros mencionan que el autor no ha liberado parte de los scripts para en entrenamiento multilenguaje.

[Notebook](https://colab.research.google.com/drive/1jLoCu8y1CyTbkORMAQv54MxmmaDtAbuy?usp=sharing)

---

### Qwen3TTS

Repositorio:  
https://github.com/QwenLM/Qwen3-TTS

Un modelo mucho más reciente y con alto reconocimiento al ser desarrollado de manera abierta por Alibaba Cloud.

Presenta varias características relevantes, la principal es el **clonamiento de voz con un audio de 3s, en cualquier idioma**.

El problema es el “peso del modelo”, tiene tantos parámetros (incluso la versión liviana) que en Google Colab con **CPU tarda cerca de 2min 40s** en generar un audio de **7s**, mientras que con **GPU tarda mínimo 35s** para el mismo audio. A pesar de ello la calidad del audio generado con CPU es impresionante.

Se intentó también reducir el tiempo haciendo el modelo solo para una voz (con fine-tunning), pero tanto en Colab como en Kaggle los tiempos no son suficientes como para completar el entrenamiento.

[Notebook](https://colab.research.google.com/drive/1v56S5mxD1bUjv21ndzFOqXQgONc6Uo7n?usp=sharing)

**Resultado**

  <video controls width="300" height="50">
    <source src="ejemplos/qwen3.wav" type="audio/wav">
  </video>
---

### PipperTTS

Repositorio:  
https://github.com/OHF-Voice/piper1-gpl

Se probó por su velocidad según comentarios en internet. Fue el modelo que menos errores presentó durante la instalación y entrenamiento.

El problema apareció al finalizar, ya que al presentar baja cantidad de parámetros (modelo rápido y liviano) los resultados son bastante deficientes, probablemente por la necesidad de un mayor tiempo de entrenamiento, algunos comentarios en internet hablan de semanas lo cual es inviable en los entornos en la nube de forma gratuita.

Durante las pruebas se notó que cada **“eppoch” tarda 26s en Kaggle con GPU P100 (~7h por cada 1000 eppochs)**, un modelo completo y funcional requiere hasta **10000 eppochs (~70h)**.

[Notebook](https://www.kaggle.com/code/luissantiagobr/pipertts)

**Resultado**

  <video controls width="300" height="50">
    <source src="ejemplos/pipper.wav" type="audio/wav">
  </video>
---

### coquiTTS con XTTSv2

Repositorio:  
https://github.com/idiap/coqui-ai-TTS

Es el único modelo relativamente funcional que fue probado. Aunque la instalación fue compleja y se considera obsoleto, se usó una versión reciente creada por la comunidad.

Es una especie de modelo de inferencia que utiliza otros modelos TTS y vocoder para generar las voces, la prueba realizada fue con XTTSv2 (de los principales y con mejor documentación).

El entrenamiento tomó varias horas, pero aprovecha un checkpoint del modelo original para no perder tiempo “configurándose” para el idioma (incluye varios idiomas), quiere decir que a diferencia de pipperTTS con 10000 eppoch, este requiere solo 100 eppoch para un resultado estable (lo utilizado) y hasta 500 eppoch para un buen entrenamiento (aunque se utilizaron algunos parámetros distintos es realista comparar los tiempos).

El modelo en sí es un clonador de voz, o sea que funciona sin el entrenamiento, pero tras algunas pruebas (meramente cualitativas) se notó que no es muy consistente, por lo que el entrenamiento es recomendable y produce buenos resultados.

El principal problema hallado es la necesidad mayor calidad, se cree que es posible resolverlo con un entrenamiento de al menos 10h, sin embargo, como se puede observar en el notebook, lo obtenido es relativamente consistente.

Por otro lado, un inconveniente es la cantidad de palabras, en sí el modelo está “bloqueado” a 259 palabras en español (aumentable manualmente desde el archivo ``), sin embargo, la razón de esto es que suele fallar tras este límite, por lo que textos largos deberían ser producidos a trozos.

Sobre la salida, se intentó obtenerla en streaming pero al ser entornos en la nube no fue posible, además se presentaron problemas de incompatibilidad con algunas dependencias del proyecto. Si se obtuvo una salida en un archivo formato `.wav` con muestreo de **24kHz**.

El tiempo para generar un audio de **7s** (misma entrada que Qwen3) con **GPU es de ~5s** y con **CPU ~28s**.

[Notebook](https://www.kaggle.com/code/luissantiagobr/coquitts-gpu)

**Resultado**

  <video controls width="300" height="50">
    <source src="ejemplos/xttsv2.wav" type="audio/wav">
  </video>

---

## Comparación

| Modelo | Instalación y Uso | Entrenamiento | Calidad | Viabilidad |
|------|------|------|------|------|
| SopranoTTS | Complejo | Fallido | No evaluada | Baja |
| Qwen3TTS | Sencillo | No evaluado | Muy alta | Baja |
| PiperTTS | Moderado | Muy largo | Baja | Media |
| CoquiTTS (XTTSv2) | Moderado | Moderada | Aceptable | Alta |