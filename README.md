# Modelos Predictivos para la Clasificación de Fases de Alzheimer mediante Aprendizaje Automático

![Universidad de Sevilla](https://img.shields.io/badge/Universidad-Sevilla-red.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Descripción del Trabajo
Este repositorio contiene el código fuente, la memoria completa, la presentación y las figuras utilizadas en el **Trabajo de Fin de Grado (TFG)** titulado:

> **"Modelos predictivos para la clasificación de fases de Alzheimer mediante técnicas de aprendizaje automático"**

* **Autor:** Cristóbal Delgado López
* **Tutores:** Luis Valencia Cabrera y Agustín Riscos Núñez
* **Titulación:** Grado en Estadística, Universidad de Sevilla

---

## 🎯 Objetivos del Proyecto

La detección temprana de la enfermedad de Alzheimer y el estadiaje preciso del deterioro cognitivo son fundamentales para mejorar el pronóstico y tratamiento de los pacientes. Este proyecto aborda dos tareas clave:

1. **Cribado Binario (Sano vs. Alzheimer / Deterioro Cognitivo)**: Determinación automatizada de la presencia o ausencia de la enfermedad para una rápida detección temprana.
2. **Estadiaje Multiclase / Regresión Ordinal (CDR: Clinical Dementia Rating)**: Clasificación de la gravedad del deterioro cognitivo (Sano $0.0$, Deterioro Leve $0.5$, Avanzado $1.0$) mediante modelos con pérdidas ordinales adaptadas.

---

## 🔬 Metodología y Arquitectura

El proyecto desarrolla una arquitectura híbrida de **fusión multimodal** diseñada para trabajar eficientemente bajo regímenes de datos reducidos (*Small Data*) evitando la fuga de datos (*Data Leakage*):

```
+------------------------------------+     +-----------------------------------+
| Resonancia Magnética (RMI) 2.5D    |     | Datos Tabulares / Psicométricos   |
| (16 cortes estructurales 224x224)  |     | (nWBV, MMSE, Edad, Educación)     |
+------------------------------------+     +-----------------------------------+
                  |                                          |
                  v                                          v
   Backbone ConvNeXt + SPD-Conv                     Procesamiento Tabular
                  |                                          |
                  +--------------------+---------------------+
                                       |
                                       v
                     Fusión Multimodal & Slice Attention
                                       |
                                       v
                    Cabeceras de Clasificación / Ordinal
```

* **Procesamiento RMI 2.5D**: Extracción de características morfológicas volumétricas a través de 16 cortes estructurales de resonancia magnética mediante un backbone **ConvNeXt** combinado con capas de descenso de resolución de conservación espacial (**SPD-Conv**) y mecanismos de atención por corte (**Slice Attention**).
* **Fusión Multimodal**: Integración de descriptores visuales con variables clínicas tabulares (volumen cerebral normado `nWBV`, edad, escolaridad) y test psicométrico **MMSE** (*Mini-Mental State Examination*).
* **Validación Cruzada Aislada por Paciente**: Esquema de validación cruzada ($K$-Fold) con estricto aislamiento a nivel de sujeto para prevenir cualquier tipo de sesgo o sobreajuste.

---

## 📊 Resultados Principales

* **Clasificación Binaria**: Rendimientos sobresalientes de **AUC > 0.95** en configuraciones multimodales completas, demostrando la alta efectividad del mecanismo *Slice Attention* y la integración de variables clínicas.
* **Estadiaje Multiclase / Ordinal**: Reducción significativa del Error Absoluto Medio (MAE) y mejora del coeficiente Quadratic Weighted Kappa (QWK) aplicando pérdidas ordinales adaptadas al espacio de etiquetas CDR.
* **Interpretabilidad**:
  * **Grad-CAM**: Identificación y visualización de regiones cerebrales críticas (atrofia hipocampal y ventricular) focalizadas por la red neuronal.
  * **SHAP (SHapley Additive exPlanations)**: Análisis del impacto y contribución de cada variable clínica y psicométrica en las predicciones.

---

## 📁 Estructura del Repositorio

```
.
├── README.md                                 # Descripción principal del proyecto
├── codigo/                                   # Código fuente en Jupyter Notebooks
│   ├── notebook_analisis.ipynb               # Análisis exploratorio y preprocesamiento de datos
│   ├── notebook_binario.ipynb                # Entrenamiento y evaluación del modelo binario
│   └── notebook_multiclase.ipynb             # Entrenamiento y evaluación del modelo multiclase/ordinal
├── imagenes/                                 # Figuras y gráficos del trabajo citados en la memoria
│   ├── binaria/                              # Gráficos y resultados del modelo binario (ROC, PR, Confusion, Grad-CAM, SHAP)
│   ├── multiclase/                           # Gráficos y resultados del modelo multiclase (ROC, PR, Loss, Boxplots, Grad-CAM, SHAP)
│   └── descriptivo/                          # Visualizaciones del análisis exploratorio (Nulos, Edades, Correlación)
└── documentacion/                            # Documentos finales del TFG
    ├── Memoria_TFG_Cristobal_Delgado.pdf     # Memoria completa en formato PDF
    └── Modelos Predictivos de Alzheimer.pdf  # Presentación diapositivas del TFG
```

---

## ⚙️ Requisitos e Instalación

Para replicar los experimentos o hacer uso del código, se recomienda contar con Python 3.9+ y PyTorch con soporte CUDA:

```bash
git clone https://github.com/delgadocristobal29/TFG-Clasificacion-de-Alzheimer.git
cd TFG-Clasificacion-de-Alzheimer
pip install -r requirements.txt
```

### Librerías Clave:
* `torch` / `torchvision`
* `numpy`, `pandas`, `scipy`
* `scikit-learn`
* `matplotlib`, `seaborn`
* `shap`
* `opencv-python`

---

## ✒️ Cita / Referencia

Si utilizas este trabajo o parte de su código en tus investigaciones, por favor cita la memoria:

```bibtex
@mastersthesis{delgado2024alzheimer,
  author       = {Delgado López, Cristóbal},
  title        = {Modelos predictivos para la clasificación de fases de Alzheimer mediante técnicas de aprendizaje automático},
  school       = {Universidad de Sevilla},
  year         = {2024},
  type         = {Trabajo de Fin de Grado}
}
```
