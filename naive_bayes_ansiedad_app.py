# Streamlit app - Sistema experto para clasificación de ansiedad académica con Naive Bayes + conexión a Airtable
import streamlit as st
import math
import requests
import datetime

# Atributos descriptivos más comprensibles para el usuario
atributos = [
    "Cuando tengo que dar una conferencia o exposición pienso que no tengo el conocimiento suficiente y que se van a burlar de mí por lo que voy a decir.",
    "Cuando expongo me da miedo que me hagan preguntas porque creo que no voy a saber qué responder, aunque sí haya estudiado el tema.",
    "Cuando participo en clase pienso que mi pregunta no es válida porque es una tontería",
    "Cuando expongo ante mi clase pienso que no me ponen atención porque carezco de autoridad ante mis compañeros.",
    "Cuando hay actividades de conversación grupal pienso excesivamente si mi argumento está bien o mal.",
    "Al convivir con compañeros pienso que se burlan de mí por lo que digo.",
    "Cuando voy a exponer ante mi clase pienso que voy a hacer el ridículo por mi manera de presentar el tema.",
    "Cuando participo en mesas de diálogo o debates en clase, suelo pensar que no me doy a entender.",
    "Suelo pensar que mis compañeros de clase se burlan de mí al participar.",
    "Cuando voy a hacer un examen oral suelo creer que no me voy a dar a entender.",
    "Suelo pensar que mis compañeros de equipo tienen un mejor uso de la palabra y por eso evito opinar.",
    "Suelo pensar que si participo en clase mis comentarios o argumentos estarán equivocados.",
    "Cuando trabajo en equipo suelo pensar que no tengo el suficiente conocimiento para expresar mi opinión sobre el tema.",
    "En los trabajos en equipo pienso que nadie me va a aceptar porque mis conocimientos son pocos, aunque apruebe los exámenes.",
    "Creo que el profesor se va a enojar conmigo después de participar porque mis participaciones son tontas.",
    "Si durante una exposición mis compañeros están platicando, pienso que soy incapaz de llamar su atención.",
    "Cuando me toca dar una conferencia evito preguntarle al público si tiene dudas por miedo a no saber qué responder",
    "Cuando estoy en una conferencia me da miedo levantar la mano para preguntar alguna duda al ponente.",
    "En los trabajos en equipo pienso que soy el que menos sabe, por lo mismo evito expresar mis ideas.",
    "Cuando voy a hacer un examen oral me da miedo no saber acomodar mis palabras para responder.",
    "Cuando tengo que dar una conferencia, exposición o simposio me da miedo pensar que el público crea que soy un charlatán por lo que estoy diciendo.",
    "Cuando tengo que dar una conferencia, exposición o simposio pienso en que me harán preguntas que no podré responder.",
    "Suelo pensar que mis conversaciones no son tan divertidas como la de mis compañeros.",
    "En los trabajos en equipo pienso que, aunque tengo suficientes conocimientos sobre el tema, los demás integrantes no me van a tomar en cuenta",
    "Suelo disminuir el volumen de mi voz por nervios cuando doy mi opinión en una clase.",
    "Cuando estoy en clase me da miedo levantar la mano para participar.",
    "Cuando convivo con mis compañeros siento que no soy parte del grupo.",
    "Creo que mis compañeros me evalúan negativamente cuando doy mi opinión en clase.",
    "Cuando voy a presentar un examen oral pienso que no sé nada, aunque sí haya estudiado.",
    "Cuando platico con mis compañeros me da miedo que no me pongan atención."
]

# Parámetros del modelo Naive Bayes ajustados a los 30 ítems (valores ficticios para ilustración)
parametros = {
    "Alto": {
        "prior": 0.33,
        "media": [3.28, 3.79, 3.35, 3.35, 3.58, 3.0, 3.18, 3.37, 3.08, 3.17, 3.26, 3.21, 2.96, 2.85, 2.91, 3.15, 3.33, 3.30, 2.96, 3.54, 2.88, 3.01, 3.0, 2.56, 2.71, 3.03, 3.24, 2.96, 3.22, 2.86],
        "desv":  [0.73, 0.45, 0.70, 0.70, 0.52, 1.22, 0.82, 0.62, 0.74, 0.98, 0.89, 0.65, 1.06, 1.20, 1.05, 1.05, 0.77, 0.79, 1.00, 0.76, 1.25, 1.10, 1.24, 1.28, 1.15, 0.93, 0.94, 1.08, 1.03, 1.19]
    },
    "Normal": {
        "prior": 0.33,
        "media": [1.39, 1.52, 1.21, 1.43, 1.88, 1.09, 1.26, 1.77, 1.62, 2.09, 1.38, 1.69, 1.18, 0.96, 1.01, 1.20, 1.79, 1.84, 0.86, 2.13, 0.75, 1.47, 1.38, 0.96, 1.66, 1.88, 1.57, 1.50, 1.84, 1.39],
        "desv":  [0.91, 0.96, 1.01, 1.00, 0.92, 0.95, 0.89, 0.76, 0.91, 0.83, 1.20, 0.81, 0.84, 1.00, 0.85, 0.93, 1.01, 1.10, 1.01, 0.97, 0.82, 1.00, 0.99, 1.08, 0.75, 0.81, 1.26, 0.96, 1.07, 1.18]
    },
    "Bajo": {
        "prior": 0.33,
        "media": [0.09, 0.37, 0.11, 0.24, 0.69, 0.0, 0.26, 0.67, 0.35, 0.52, 0.07, 0.67, 0.26, 0.0, 0.17, 0.30, 0.45, 0.52, 0.11, 1.01, 0.17, 0.41, 0.54, 0.11, 0.56, 0.98, 0.66, 0.67, 0.83, 0.32],
        "desv":  [0.35, 0.65, 0.31, 0.43, 1.20, 0.17, 0.64, 0.86, 0.55, 0.96, 0.26, 0.63, 0.44, 0.17, 0.37, 0.66, 0.81, 0.63, 0.31, 0.94, 0.50, 0.76, 0.90, 0.31, 0.87, 0.78, 0.86, 0.84, 0.84, 0.60]
    }
}

def probabilidad_gaussiana(x, mu, sigma):
    if sigma == 0:
        return 1.0 if x == mu else 1e-9
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(- ((x - mu) ** 2) / (2 * sigma ** 2))

def clasificar_naive_bayes(respuestas):
    log_probabilidades = {}
    for clase, stats in parametros.items():
        logp = math.log(stats["prior"])
        for i, valor in enumerate(respuestas):
            mu = stats["media"][i]
            sigma = stats["desv"][i]
            p = probabilidad_gaussiana(valor, mu, sigma)
            logp += math.log(p if p > 0 else 1e-9)
        log_probabilidades[clase] = logp
    clase_predicha = max(log_probabilidades, key=log_probabilidades.get)
    return log_probabilidades, clase_predicha

def guardar_en_airtable(respuestas, clase):
    url = f"https://api.airtable.com/v0/{st.secrets['AIRTABLE_BASE_ID']}/{st.secrets['AIRTABLE_TABLE_NAME']}"
    headers = {
        "Authorization": f"Bearer {st.secrets['AIRTABLE_TOKEN']}",
        "Content-Type": "application/json"
    }
    fields = {"Fecha": str(datetime.date.today()), "Clase": clase}
    for i, atributo in enumerate(atributos):
        fields[f"item_{i+1}"] = respuestas[i]
    data = {"records": [{"fields": fields}]}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        st.error(f"❌ Error al guardar en Airtable: {response.status_code} - {response.text}")
    else:
        st.info("📝 Las respuestas han sido guardadas en Airtable correctamente.")

st.title("🔍 Sistema Experto: Clasificación de Ansiedad Académica")
st.write("Responde cada reactivo del cuestionario con un valor de 0 (muy en desacuerdo) a 5 (muy de acuerdo).")

respuestas_usuario = []

with st.form("cuestionario"):
    for atributo in atributos:
        val = st.slider(atributo, 0, 5, 3)
        respuestas_usuario.append(val)
    submitted = st.form_submit_button("Clasificar")

if submitted:
    logps, clase = clasificar_naive_bayes(respuestas_usuario)

    st.subheader("📊 Resultados")
    for clase_nombre, logp in logps.items():
        st.write(f"**{clase_nombre}**: log-prob = {logp:.4f}")

    st.success(f"✅ Clasificación final: **{clase}**")

    guardar_en_airtable(respuestas_usuario, clase)

    st.markdown("""
    ---
    #### ℹ️ ¿Qué significa *log-prob*?
    El modelo usa logaritmos para calcular probabilidades de forma más estable.
    El valor más cercano a cero (menos negativo) indica la clase más probable.
    """)
