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

# (parámetros estadísticos omitidos aquí por brevedad, pero se deben mantener sin cambios si el modelo no cambia)

# (funciones de clasificación también sin cambios...)

# Reemplazar st.info por un manejo de errores explícito

def guardar_en_airtable(respuestas, clase):
    url = f"https://api.airtable.com/v0/{st.secrets['AIRTABLE_BASE_ID']}/{st.secrets['AIRTABLE_TABLE_NAME']}"
    headers = {
        "Authorization": f"Bearer {st.secrets['AIRTABLE_TOKEN']}",
        "Content-Type": "application/json"
    }
    fields = {"Fecha": str(datetime.date.today()), "Clase": clase}
    for i, atributo in enumerate(atributos):
        fields[f"item_{i+1}"] = respuestas[i]  # guardamos como item_1, item_2...
    data = {"records": [{"fields": fields}]}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        st.error(f"❌ Error al guardar en Airtable: {response.status_code} - {response.text}")
    else:
        st.info("📝 Las respuestas han sido guardadas en Airtable correctamente.")

# Interfaz de usuario
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
