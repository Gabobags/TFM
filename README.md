# Airbnb Pricing Intelligence NYC

Este proyecto corresponde a una demo de productivización de un modelo de Machine Learning para estimar precios de alojamientos Airbnb en Nueva York.

## 📦 Contenido
- streamlit_app_best.py → aplicación principal
- AB_NYC_2019.csv → dataset 2019
- AB_NYC_2024.csv → dataset 2024
- requirements.txt → dependencias
- README.md → instrucciones

## 🚀 Cómo ejecutar

1. Abrir una terminal en esta carpeta

2. Instalar dependencias:
pip install -r requirements.txt

3. Ejecutar la aplicación:
streamlit run streamlit_app_best.py

4. Abrir en el navegador:
http://localhost:8501

## 💡 Descripción

La aplicación permite:
- estimar precios basados en 2019
- estimar precios basados en 2024
- comparar ambos modelos
- obtener una estimación actual basada en el dataset más reciente

## 🧠 Nota metodológica

La estimación “actual” no corresponde a datos en tiempo real, sino a una aproximación basada en el dataset más reciente disponible (2024).
