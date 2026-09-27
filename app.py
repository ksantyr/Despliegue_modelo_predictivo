import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Cargar modelo
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("modelo-rn.pkl", "rb") as f:
        bundle = pickle.load(f)
    model     = bundle[0]   # KNeighborsClassifier
    label_enc = bundle[1]   # LabelEncoder (Aprobado)
    col_names = bundle[2]   # columnas esperadas por el modelo
    scaler    = bundle[3]   # MinMaxScaler (solo float64)
    return model, label_enc, col_names, scaler

model, label_enc, col_names, scaler = load_model()

NUM_COLS = [
    "Edad", "Créditos Académicos",
    "creditos_aprobados_hist", "creditos_reprobados_hist",
    "tasa_aprobacion_hist", "promedio_notas_hist",
    "tasa_aprobacion_materia_hist", "tasa_aprobacion_docente_hist",
    "Edad_profesor",
]

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Predictor de Aprobación", layout="wide")
st.title("Predicción de Aprobación de Curso")
st.markdown("Completa los datos del estudiante, el curso y el docente para obtener la predicción.")

with st.form("formulario"):
    st.subheader("Datos del estudiante")
    c1, c2, c3 = st.columns(3)
    with c1:
        edad        = st.number_input("Edad", 15, 70, 22)
        genero      = st.selectbox("Género", ["F", "M"])
        estrato     = st.selectbox("Estrato", ["1", "2", "3", "4", "5", "6", "Sin dato"])
    with c2:
        programa    = st.selectbox("Programa", [
            "Admón de Empresas-Med", "Economía - Med",
            "Gestión del empren y la innova", "Negocios Internal-Med",
        ])
        tipo_alumno = st.selectbox("Tipo Alumno", [
            "C - Estudiante matriculado", "F - Transferencia interna",
            "J - Transferencia externa", "K - Convenio", "N - Nuevo",
            "R - Reintegro", "X - Transitorio", "Y - Transformación curricular",
        ])
        ubicacion   = st.selectbox("Ubicación", [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
    with c3:
        ano_saber11        = st.selectbox("Año Saber 11", [
            2014.0,2015.0,2016.0,2017.0,2018.0,2019.0,
            2020.0,2021.0,2022.0,2023.0,2024.0,2025.0,
        ])
        naturaleza_colegio = st.selectbox("Naturaleza Colegio", ["OFICIAL", "PRIVADO"])
        origen_colegio     = st.selectbox("Origen Colegio", ["local", "no local"])
        origen_nacimiento  = st.selectbox("Origen Nacimiento", ["local", "no local"])
        tipo_municipio     = st.selectbox("Tipo Municipio Nacimiento", [
            "Ciudad / aglomeración", "Intermedio", "Rural",
        ])

    st.divider()
    st.subheader("Historial académico del estudiante")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        creditos_aprobados_hist  = st.number_input("Créditos aprobados (hist.)", 0.0, 300.0, 40.0, step=1.0)
    with h2:
        creditos_reprobados_hist = st.number_input("Créditos reprobados (hist.)", 0.0, 100.0, 0.0, step=1.0)
    with h3:
        tasa_aprobacion_hist     = st.slider("Tasa aprobación histórica", 0.0, 1.0, 0.85)
    with h4:
        promedio_notas_hist      = st.number_input("Promedio notas histórico", 0.0, 5.0, 3.5, step=0.1)

    st.divider()
    st.subheader("Datos del curso")
    cu1, cu2, cu3 = st.columns(3)
    with cu1:
        creditos_academicos      = st.number_input("Créditos Académicos", 1, 10, 3)
        metodo_asistencia        = st.selectbox("Método Asistencia", [
            "A - A distancia", "C - Clase presencial", "D - Curso Dirigido",
            "G - Requisito de Grado", "P - Curso proyecto",
            "T - Telepresencial", "V - Virtual", "Z - Ciclo de integración",
        ])
    with cu2:
        categoria_materia        = st.selectbox("Categoría Materia", [
            "Ciencia Básica", "Electivas / Optativas",
            "Formación Básica Profesional", "Formación Humanística Institucional",
            "Formación Profesional Específica", "Investigación", "Práctica",
        ])
    with cu3:
        tasa_aprobacion_materia_hist = st.slider("Tasa aprobación materia (hist.)", 0.0, 1.0, 0.80)

    st.divider()
    st.subheader("Datos del docente")
    d1, d2, d3 = st.columns(3)
    with d1:
        edad_profesor            = st.number_input("Edad del docente", 25, 80, 45)
        nivel_formacion          = st.selectbox("Nivel Formación", [
            "005-Formacion Tecnica", "007-Universitaria",
            "009-Esp. Tecnologica", "010-Esp. Universitaria",
            "012-Maestria", "013-Doctorado", "014-Postdoctorado",
        ])
        tasa_aprobacion_docente_hist = st.slider("Tasa aprobación docente (hist.)", 0.0, 1.0, 0.85)
    with d2:
        escuela                  = st.selectbox("Escuela", [
            "Arquitectura y Diseño", "CIDI - UPB", "Centro de Lenguas",
            "Ciencias Sociales", "Ciencias de la Salud", "Colegio",
            "Derecho y Ciencias Políticas",
            "Economía, Administración y Negocios",
            "Educación y Pedagogía", "Ingenierías", "Otros",
            "Teología, Filosofía y Humanidades",
        ])
        rol                      = st.selectbox("Rol", [
            "Académico", "Apoyo Y Soporte", "Estratégico",
            "Sin Asignar Rol", "Táctico", "Táctico-Estratégico",
        ])
    with d3:
        tipo_empleado            = st.selectbox("Tipo Empleado", [
            "DCD - Docente Comisión Administrativ",
            "DEF - Docente Externo de Facultad",
            "DIC - Docente Interno de Colegio",
            "DIF - Docente Interno de Facultad",
            "DIP - Docente Interno de Posgrado",
            "DPM - Prestación  Servicios Med",
            "IAD - Interno Administrativo",
        ])
        vinculacion              = st.selectbox("Vinculación", ["Externo", "Interno"])

    submitted = st.form_submit_button("Predecir", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
if submitted:
    # Inicializar fila con ceros
    row = {c: 0 for c in col_names}

    # --- Numéricas ---
    row["Edad"]                          = float(edad)
    row["Créditos Académicos"]           = float(creditos_academicos)
    row["creditos_aprobados_hist"]       = float(creditos_aprobados_hist)
    row["creditos_reprobados_hist"]      = float(creditos_reprobados_hist)
    row["tasa_aprobacion_hist"]          = float(tasa_aprobacion_hist)
    row["promedio_notas_hist"]           = float(promedio_notas_hist)
    row["tasa_aprobacion_materia_hist"]  = float(tasa_aprobacion_materia_hist)
    row["tasa_aprobacion_docente_hist"]  = float(tasa_aprobacion_docente_hist)
    row["Edad_profesor"]                 = float(edad_profesor)

    # --- Dummies multi-categoría ---
    for col_key, value in [
        ("Año Saber11",              ano_saber11),
        ("Estrato",                  estrato),
        ("Método Asistencia Curso",  metodo_asistencia),
        ("Tipo Alumno",              tipo_alumno),
        ("Ubicación",                ubicacion),
        ("Programa1",                programa),
        ("Escuela",                  escuela),
        ("Nivel Formación",          nivel_formacion),
        ("Rol",                      rol),
        ("Tipo Empleado",            tipo_empleado),
        ("Categoria_Materia",        categoria_materia),
        ("Tipo_Municipio_Nacimiento", tipo_municipio),
    ]:
        key = f"{col_key}_{value}"
        if key in row:
            row[key] = 1

    # --- Dummies binarias (drop_first=True → solo el 2.º nivel tiene columna) ---
    if naturaleza_colegio == "OFICIAL":
        row["Naturaleza Colegio_OFICIAL"] = 1
    if genero == "M":
        row["Género_M"] = 1
    if vinculacion == "Interno":
        row["Vinculación_Interno"] = 1
    if origen_nacimiento == "local":
        row["ORIGEN_NACIMIENTO_local"] = 1
    if origen_colegio == "local":
        row["ORIGEN_COLEGIO_local"] = 1

    # --- DataFrame y escalado ---
    X_input = pd.DataFrame([row], columns=col_names).astype(float)
    X_input[NUM_COLS] = scaler.transform(X_input[NUM_COLS])

    # --- Predicción ---
    pred  = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    # Las clases del modelo son los labels entrenados; clase 1 = Aprobado
    idx_aprobado   = list(model.classes_).index(1) if 1 in model.classes_ else 1
    prob_aprobado  = proba[idx_aprobado]
    prob_reprobado = 1 - prob_aprobado

    st.divider()
    if pred == 1:
        st.success(f"### El estudiante **APRUEBA** el curso")
    else:
        st.error(f"### El estudiante **REPRUEBA** el curso")

    col_p, col_r = st.columns(2)
    col_p.metric("Probabilidad de aprobar",   f"{prob_aprobado:.1%}")
    col_r.metric("Probabilidad de reprobar",  f"{prob_reprobado:.1%}")

    st.warning(
        "**Rendimiento del modelo — clase Reprobado (0):**  "
        "Precisión = 62.5 % · Recall = 51.7 % · F1 = 56.5 %"
    )

    with st.expander("Ver detalle del vector de entrada"):
        st.dataframe(
            pd.DataFrame({"Columna": col_names, "Valor": X_input.iloc[0].values}),
            use_container_width=True,
        )
