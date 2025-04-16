# Streamlit app - Sistema experto para clasificación de ansiedad académica con Naive Bayes + conexión a Airtable
import streamlit as st
import math
import requests
import datetime

# Mapeo de campos cortos a enunciados largos para la visualización
campos_airtable = {f"item_{i+1}": texto for i, texto in enumerate([
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
])}

atributos = list(campos_airtable.keys())

parametros = {  # ... igual que antes (omitido aquí para brevedad)
    "Alto": {"prior": 0.33, "media": [...], "desv": [...]},
    "Normal": {"prior": 0.33, "media": [...], "desv": [...]},
    "Bajo": {"prior": 0.33, "media": [...], "desv": [...]}
}

# Funciones para clasificación y guardar en Airtable (igual que antes)

st.title("🔍 Sistema Experto: Clasificación de Ansiedad Académica")
st.write("Responde cada reactivo del cuestionario con un valor de 0 (muy en desacuerdo) a 5 (muy de acuerdo).")

respuestas_usuario = []

with st.form("cuestionario"):
    for clave, texto in campos_airtable.items():
        val = st.slider(texto, 0, 5, 3)
        respuestas_usuario.append(val)
    submitted = st.form_submit_button("Clasificar")

if submitted:
    logps, clase = clasificar_naive_bayes(respuestas_usuario)

    st.subheader("📊 Resultados")
    for clase_nombre, logp in logps.items():
        st.write(f"**{clase_nombre}**: log-prob = {logp:.4f}")

    st.success(f"✅ Clasificación final: **{clase}**")

    # Mostrar perfil personalizado
    if clase == "Alto":
        st.markdown("""
        ### 🛑 Perfil: Nivel *Alto*
        **Factor: Ansiedad por Participación Oral / Expresión Verbal**

        Tu perfil indica un nivel **alto** de ansiedad asociada con situaciones en las que necesitas hablar en público, participar en clase o expresar tus ideas.

        - Tiendes a pensar que tus preguntas u opiniones no son válidas o que los demás se burlarán de ti.  
        - Evitas participar o preguntar por miedo a no ser entendido o a equivocarte.  
        - Las exposiciones, debates o conferencias generan en ti un temor significativo, incluso si has preparado el tema.  
        - Percibes que careces de autoridad o que no podrás responder preguntas, lo cual afecta tu confianza.  
        - Puedes sentir que los profesores o compañeros te juzgan negativamente.

        🧠 **Sugerencia:** Este nivel de ansiedad puede interferir con tu rendimiento académico. Sería útil trabajar estrategias de afrontamiento, práctica controlada de exposición oral y posiblemente acompañamiento psicológico para reducir estas percepciones y mejorar tu seguridad al hablar.
        """)
    elif clase == "Normal":
        st.markdown("""
        ### 🟡 Perfil: Nivel *Normal*
        **Factor: Ansiedad por Participación Oral / Expresión Verbal**

        Tu perfil indica un nivel **moderado** o **normal** de ansiedad en contextos de participación oral.

        - Puedes sentir nervios o inseguridad en situaciones sociales o académicas, pero generalmente puedes afrontarlas.  
        - Es posible que ocasionalmente dudes de tus respuestas o evites hablar en público, pero no de forma constante.  
        - El miedo a ser evaluado existe, pero no paraliza tu participación.

        💡 **Sugerencia:** Puedes beneficiarte de seguir practicando la expresión oral, reforzando tu confianza y exponiéndote a estos contextos poco a poco.
        """)
    else:
        st.markdown("""
        ### 🟢 Perfil: Nivel *Bajo*
        **Factor: Ansiedad por Participación Oral / Expresión Verbal**

        Tu perfil indica un nivel **bajo** de ansiedad en situaciones que implican participación verbal.

        - Te sientes cómodo expresando tus ideas en clase, en exposiciones o debates.  
        - No temes equivocarte ni ser evaluado negativamente por tus compañeros o profesores.  
        - Sueles confiar en tu conocimiento y no te preocupa en exceso lo que piensen los demás.

        🌟 **Sugerencia:** Este es un perfil muy favorable. Puedes usar tu seguridad y habilidades comunicativas para apoyar a otros compañeros, participar activamente y convertirte en un líder dentro de tu grupo académico.
        """)

    guardar_en_airtable(respuestas_usuario, clase)

    st.markdown("""
    ---
    #### ℹ️ ¿Qué significa *log-prob*?
    El modelo usa logaritmos para calcular probabilidades de forma más estable.
    El valor más cercano a cero (menos negativo) indica la clase más probable.
    """)
