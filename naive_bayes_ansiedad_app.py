
# Streamlit app – Sistema Experto de Ansiedad Académica (modelo Naive Bayes optimizado)
import streamlit as st
import math

atributos = [
    "F1_conferencias_pienso_conocimiento_burlar",
    "F1expongo_miedo_preguntas_responder",
    "F1participacion_pregunta_no_valida",
    "F1exposicion_pienso_carezco_autoridad",
    "F1dialogo_actividades_excesivamente_argumento",
    "F1social_companeros_burlar_digo",
    "F1exposicion_clase_presentar_tema",
    "F1dialogo_participo_mesas_entender",
    "F1participacion_pensar_companeros_burlan",
    "F1examen_creer_dar_entender",
    "F1equipo_pensar_companeros_evito",
    "F1participacion_pensar_participo_equivocado",
    "F1equipo_pensar_suficiente_opinion",
    "F1equipo_aceptar_conocimientos_examenes",
    "F1participacion_profesor_enojar_participaciones",
    "F1exposicion_platicando_incapaz_atencion",
    "F1conferencias_pienso_preguntas_responder",
    "F1conferecia_miedo_levantar_mano",
    "F1equipo_pienso_evito_ideas",
    "F1examen_miedo_acomodar_palabras",
    "F1conferencias_miedo_publico_charlatan",
    "F1conferencia_publico_miedo_responder",
    "F1social_conversaciones_divertidas_companeros",
    "F1equipo_pienso_conocimientos_integrantes",
    "F1participacion_disminuir_voz_nervios_opinion",
    "F1clase_miedo_mano_participar",
    "F1social_convivo_parte_grupo",
    "F1participacion_companeros_evaluan_negativamente",
    "F1examen_presentar_pienso_estudiado",
    "F1social_platico_miedo_atencion"
]

parametros = {
    "Alto": {
        "prior": 0.33,
        "media": [3.283, 3.7925, 3.3585, 3.3585, 3.5849, 3.0, 3.1887, 3.3774, 3.0755, 3.1698, 3.2642, 3.2075, 2.9623, 2.8491, 2.9057, 3.1509, 3.3396, 3.3019, 2.9623, 3.5472, 2.8868, 3.0189, 3.0, 2.566, 2.717, 3.0377, 3.2453, 2.9623, 3.2264, 2.8679],
        "desv": [0.7366, 0.4497, 0.7029, 0.7029, 0.5296, 1.2286, 0.8255, 0.6212, 0.7486, 0.9855, 0.8934, 0.6547, 1.0633, 1.2037, 1.0509, 1.0532, 0.7757, 0.7911, 1.0087, 0.7664, 1.2538, 1.1073, 1.2439, 1.2814, 1.1554, 0.9309, 0.9498, 1.0809, 1.0396, 1.1981]
    },
    "Normal": {
        "prior": 0.33,
        "media": [1.3962, 1.5283, 1.2075, 1.434, 1.8868, 1.0943, 1.2642, 1.7736, 1.6226, 2.0943, 1.3774, 1.6981, 1.1887, 0.9623, 1.0189, 1.2075, 1.7925, 1.8491, 0.8679, 2.1321, 0.7547, 1.4717, 1.3774, 0.9623, 1.6604, 1.8868, 1.566, 1.5094, 1.8491, 1.3962],
        "desv": [0.9182, 0.9636, 1.0161, 1.0002, 0.9247, 0.9569, 0.8934, 0.7683, 0.9158, 0.8302, 1.201, 0.8146, 0.848, 1.0087, 0.8576, 0.9389, 1.0161, 1.1057, 1.0101, 0.972, 0.822, 1.002, 0.9948, 1.0809, 0.7509, 0.8164, 1.2665, 0.9639, 1.071, 1.187]
    },
    "Bajo": {
        "prior": 0.33,
        "media": [0.0943, 0.3774, 0.1132, 0.2453, 0.6981, 0.0, 0.2642, 0.6792, 0.3585, 0.5283, 0.0755, 0.6792, 0.2642, 0.0, 0.1698, 0.3019, 0.4528, 0.5283, 0.1132, 1.0189, 0.1698, 0.4151, 0.5472, 0.1132, 0.566, 0.9811, 0.6604, 0.6792, 0.8302, 0.3208],
        "desv": [0.351, 0.6509, 0.3168, 0.4303, 1.2067, 0.1667, 0.6487, 0.8638, 0.5527, 0.9636, 0.2642, 0.6376, 0.4409, 0.1667, 0.3755, 0.6612, 0.8142, 0.6326, 0.3168, 0.9415, 0.5042, 0.7632, 0.9021, 0.3168, 0.8797, 0.7889, 0.8675, 0.8417, 0.8408, 0.6073]
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

st.title("🧠 Clasificador de Ansiedad Académica")
st.write("Responde cada reactivo con un valor de 0 (muy en desacuerdo) a 5 (muy de acuerdo).")

respuestas_usuario = []
with st.form("cuestionario"):
    for atributo in atributos:
        val = st.slider(atributo.replace("_", " "), 0, 5, 3)
        respuestas_usuario.append(val)
    submitted = st.form_submit_button("Clasificar")

if submitted:
    logps, clase = clasificar_naive_bayes(respuestas_usuario)
    st.subheader("📊 Resultados")
    for clase_nombre, logp in logps.items():
        st.write(f"**{clase_nombre}**: log-prob = {logp:.4f}")
    st.success(f"✅ Clasificación final: **{clase}**")

    if clase == "Alto":
        st.warning("Este perfil sugiere ansiedad académica alta. Podría interferir significativamente con el desempeño escolar y bienestar emocional.")
    elif clase == "Normal":
        st.info("Este perfil indica un nivel moderado de ansiedad, con áreas puntuales que podrían trabajarse para mejorar la participación académica.")
    elif clase == "Bajo":
        st.success("Este perfil refleja una ansiedad académica baja. Hay confianza para participar y exponerse en contextos escolares.")

    st.markdown("---")
    st.markdown("ℹ️ *El sistema utiliza inferencia bayesiana y datos empíricos para estimar tu nivel de ansiedad académica.*")
