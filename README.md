# Modelos Predictivos para la Clasificación de Fases de Alzheimer mediante Aprendizaje Automático

[![Universidad de Sevilla](https://img.shields.io/badge/Universidad-Sevilla-red.svg)](https://www.us.es/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Descripción del Trabajo

Este repositorio contiene el código fuente, la memoria académica completa, la presentación y las figuras utilizadas en el **Trabajo de Fin de Grado (TFG)** titulado:

> **"Modelos predictivos para la clasificación de fases de Alzheimer mediante técnicas de aprendizaje automático"**

*  **Autor:** Cristóbal Delgado López
*  **Tutores:** Dr. Luis Valencia Cabrera y Dr. Agustín Riscos Núñez
*  **Institución:** Universidad de Sevilla
*  **Titulación:** Grado en Estadística

---

## Objetivos del Proyecto

La detección temprana del Alzheimer y la estadificación precisa del deterioro cognitivo resultan esenciales para mejorar la intervención clínica. El proyecto aborda este desafío maximizando el aprovechamiento de fuentes de datos heterogéneas mediante una estrategia de fusión multimodal, combinando la información volumétrica de la neuroimagen estructural (sMRI) con marcadores demográficos y evaluaciones psicométricas tradicionales.  

El diseño busca obtener un sistema altamente robusto que mitigue la dependencia exclusiva de una única modalidad (Modality Laziness) y garantice la máxima honestidad metodológica y reproducibilidad clínica. Para ello, se erradican los falsos rendimientos y las métricas artificialmente infladas habituales en la literatura mediante un protocolo estricto de prevención de fuga de datos (Data Leakage) con particionado a nivel de sujeto (Subject-Level Split).  

Bajo este marco de rigor, el problema se estructura en dos enfoques predictivos complementarios:  
Cribado Binario (Sano vs. Patológico): Detección primaria de la presencia o ausencia de neurodegeneración ($CDR = 0$ vs. $CDR > 0$) para cribado y triaje inicial. 

Estadificación Multiclase / Regresión Ordinal: Proyección de la progresión continua y jerárquica de la enfermedad (Sano $0.0$, Deterioro Cognitivo Leve / MCI $0.5$, Demencia / Alzheimer $1.0$) mediante la arquitectura CORAL (Consistent Rank Logits) y decodificación por Esperanza Matemática $\mathbb{E}[y]$.  

---

## Metodología y Rigor Experimental

El proyecto implementa una arquitectura híbrida de **fusión multimodal** diseñada para operar de forma honesta bajo regímenes de datos limitados (*Small Data*, cohorte **OASIS-1** consolidada en $N=172$ pacientes), erradicando los sesgos metodológicos recurrentes en la literatura:

```text
+------------------------------------+       +-----------------------------------+
| Resonancia Magnética (mri) 2.5D    |       | Datos Tabulares / Psicométricos   |
| (16 cortes estructurales 224x224)  |       | (nWBV, MMSE, Edad, Educación)     |
+------------------------------------+       +-----------------------------------+
                  |                                            |
                  v                                            v
    Backbone ConvNeXt + SPD-Conv                     Procesamiento Tabular
                  |                                            |
                  +--------------------+-----------------------+
                                       |
                                       v
                         Fusión Multimodal & Slice Attention
                                       |
                                       v
                        Cabeceras de Clasificación / Ordinal
```

* **Control de Fuga de Datos (*Data Leakage*)**: Particionado estricto a nivel de sujeto (*Subject-Level Split*) dentro de una validación cruzada estratificada de 5 pliegues (*Stratified 5-Fold Cross-Validation*). Ningún corte de un paciente coexiste simultáneamente en entrenamiento y prueba.
* **Estratificación Dual**: Neutralización del desplazamiento de covariable (*Covariate Shift*) emparejando cuartiles de edad con el diagnóstico clínico para evitar que la red clasifique el envejecimiento natural en lugar de la atrofia patológica.
* **Procesamiento de Neuroimagen 2.5D**: Extracción espacial mediante *ConvNeXt-Tiny* modificado con convolución *Space-to-Depth* (**SPD-Conv**) para evitar la pérdida de granularidad subpíxel del *Max-Pooling*, integrando un módulo **Slice-Attention** con activación mixta (**MAF**) para ponderar los 16 planos transversales más patológicos.
* **Regularización e Inferencia**: Entrenamiento con *Focal Smooth Loss*, *Stochastic Weight Averaging* (SWA) y evaluación bajo *Test-Time Augmentation* (TTA) con 10 pasadas afines.
* **Orquestación Multimodal**:
  * **Régimen Binario**: Ensamblado en espacio *logit* mediante **LASSO Stacking** ($L_1$), seleccionando analíticamente los pesos y calibrando el punto de corte óptimo con el **Índice de Youden**.
  * **Régimen Ordinal**: Integración mediante **Mezcla Convexa Lineal** de la Esperanza Matemática visual $\mathbb{E}[y]$ y el riesgo tabular de un regresor **Bayesian Ridge** con expansión polinómica de segundo grado, discretizado por optimización **Maximin**.

---

## Resultados Principales

Todas las métricas reportadas provienen de predicciones consolidadas fuera de pliegue (**Out-Of-Fold, OOF**), garantizando una evaluación realista y libre de sobreajuste.

### 1. Régimen Binario (Sano vs. Patológico)

| Modelo / Escenario | AUC-ROC | PR-AUC | F1-Score | Sensibilidad | Especificidad |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Visual Aislada (ConvNeXt 2.5D)** | 0.7584 | 0.7153 | 0.7519 | 0.7765 | 0.7561 |
| **2. Tabular (Sin MMSE, con nWBV)** | 0.7512 | 0.7114 | 0.7404 | 0.7750 | 0.7345 |
| **3. Tabular Completo (con MMSE)** | 0.8889 | 0.8948 | 0.8559 | 0.8890 | 0.8345 |
| **4. Fusión Base (Demografía)** | 0.7448 | 0.6848 | 0.7686 | 0.8640 | 0.6579 |
| **5. Fusión Sin MMSE** | 0.7642 | 0.7166 | 0.7534 | 0.7772 | 0.7573 |
| **6. Fusión Completa (LASSO Stacking)** | **0.8937** | **0.8996** | **0.8517** | **0.8640** | **0.8567** |

* **Optimización de Youden**: La calibración del umbral global ($t_{\text{opt}} = 0.4675$, $J=0.7207$) reduce las omisiones clínicas (Falsos Negativos) en la Fusión Completa a solo 11 pacientes frente a los 18 de la rama visual aislada.
* **Capacidad de Rescate y Mitigación del Efecto Techo**: En presencia del test MMSE, el meta-modelo LASSO asigna un peso del $93.7\%$ a la tabla y un $6.3\%$ a la neuroimagen (*Modality Laziness*). Lejos de ser ruido, este $6.3\%$ actúa como **red de seguridad clínica**: el modelo visual rescata a pacientes con deterioro cognitivo incipiente (CDR 0.5) que presentaban puntuaciones perfectas en papel (MMSE 29-30/30) debido a su alta reserva cognitiva (ej. sujetos `OAS1_0042`, `OAS1_0205` y `OAS1_0263`), asignándoles puntuaciones de riesgo visual $\ge 0.50$ que superan el umbral de detección.

---

### 2. Régimen Ordinal Multiclase (Progresión Continua CORAL)

| Configuración del Modelo | AUC-ROC | PR-AUC | QWK | MAE | Peso Visual ($W_{\text{vis}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Visual Aislada (ConvNeXt-CORAL)** | 0.7171 | 0.6650 | 0.3079 | 0.3293 | 1.000 |
| **2. Tabular (Sin MMSE, con nWBV)** | 0.7225 | 0.6798 | 0.3636 | 0.2712 | 0.000 |
| **3. Tabular Completo (con MMSE)** | **0.8496** | **0.8505** | **0.5617** | **0.2145** | 0.000 |
| **4. Fusión Base (Demografía)** | 0.7131 | 0.6657 | 0.3015 | 0.3112 | 0.758 |
| **5. Fusión Sin MMSE** | 0.7412 | 0.7098 | 0.3789 | 0.2905 | **0.594** |
| **6. Fusión Completa** | **0.8496** | **0.8505** | **0.5617** | **0.2145** | 0.000 |

* **Adaptabilidad Multimodal**: En ausencia de evaluaciones neuropsicológicas (Fusión Sin MMSE), la red asume el $59.4\%$ de la inferencia combinándose con la volumetría `nWBV`, elevando el AUC a 0.7412 y mejorando la sensibilidad en fases de transición (MCI).
* **Defensa Algorítmica**: En la Fusión Completa, el optimizador anula la rama visual ($W_{\text{vis}} = 0.000$) para evitar transferencias negativas y proteger el error continuo (MAE = 0.2145) frente a la alta correlación lineal del MMSE.

---

### 3. Interpretabilidad y Validación Anatómica (XAI)

* **Grad-CAM Espacial**: Las activaciones térmicas de la red visual confirman un anclaje anatómico genuino en los marcadores de neurodegeneración (N), focalizando la atención en la ventriculomegalia (atrofia *ex-vacuo*) en el régimen binario y en el ensanchamiento del espacio extracortical (atrofia cortical difusa) en el régimen ordinal, ignorando el cráneo o artefactos del escáner.
*  **SHAP (SHapley Additive exPlanations)**: La descomposición de valores SHAP ratifica la dominancia clínica del `MMSE` y la atrofia volumétrica `nWBV`, justificando la distribución paramétrica de los meta-modelos de fusión.

---

## Estructura del Repositorio

```text
.
├── README.md                                 # Descripción principal del proyecto
├── requirements.txt                          # Dependencias de Python
├── codigo/                                   # Código fuente en Jupyter Notebooks
│   ├── notebook_analisis.ipynb               # EDA, auditoría de datos y barrido etario
│   ├── notebook_binario.ipynb                # Modelo ConvNeXt 2.5D, LASSO Stacking, Grad-CAM y métricas
│   └── notebook_multiclase.ipynb             # Modelo CORAL, Bayesian Ridge, calibración y SHAP
├── imagenes/                                 # Figuras y gráficos citados en la memoria
│   ├── binaria/                              # ROC, PR, matrices de confusión, Grad-CAM binario y SHAP
│   ├── multiclase/                           # ROC, PR, matrices 3x3, KDE, Grad-CAM ordinal y boxplots
│   └── descriptivo/                          # Auditoría de nulos, histogramas y correlación de Spearman
└── documentacion/                            # Documentos finales del TFG
    ├── Memoria_TFG_Cristobal_Delgado.pdf     # Memoria escrita completa del TFG
    └── Modelos Predictivos de Alzheimer.pdf # Diapositivas empleadas en la defensa
```

---

## Requisitos e Instalación

Para ejecutar los notebooks localmente o en entornos de GPU acelerada (Kaggle T4 / Google Colab / Local CUDA):

```bash
# Clonar el repositorio
git clone https://github.com/delgadocristobal29/TFG-Clasificacion-de-Alzheimer.git
cd TFG-Clasificacion-de-Alzheimer

# Crear y activar entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Librerías Principales

* **Deep Learning:** `torch`, `torchvision`
* **Modelado Tabular & Métricas:** `scikit-learn`
* **Optimización Hiperparamétrica:** `optuna` (TPE)
* **Procesamiento de Imagen:** `opencv-python`, `Pillow`
* **Explicabilidad (XAI) & Visualización:** `shap`, `matplotlib`, `seaborn`
* **Computación Numérica & Datos:** `numpy`, `pandas`, `scipy`

---

## Cita / Referencia

Si utilizas este trabajo o parte de su código en tus investigaciones o proyectos, por favor cita la memoria académica:

```bibtex
@mastersthesis{delgado2026alzheimer,
  author       = {Delgado López, Cristóbal},
  title        = {Modelos predictivos para la clasificación de fases de Alzheimer mediante técnicas de aprendizaje automático},
  school       = {Universidad de Sevilla},
  year         = {2026},
  type         = {Trabajo de Fin de Grado}
}
```

---

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
