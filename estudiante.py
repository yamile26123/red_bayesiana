from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Definición del grafo
model = DiscreteBayesianNetwork([
    ("HorasEstudio", "Asistencia"),
    ("Asistencia", "NotaExamen"),
    ("NotaExamen", "Aprobacion")
])

# 2. Definición de probabilidades (CPDs de la lámina)
cpd_horas = TabularCPD(
    variable="HorasEstudio",
    variable_card=2,
    values=[[0.60], [0.40]],
    state_names={"HorasEstudio": ["Pocas", "Muchas"]}
)

cpd_asistencia = TabularCPD(
    variable="Asistencia",
    variable_card=2,
    values=[[0.80, 0.30], [0.20, 0.70]],
    evidence=["HorasEstudio"],
    evidence_card=[2],
    state_names={"Asistencia": ["Baja", "Alta"], "HorasEstudio": ["Pocas", "Muchas"]}
)

cpd_nota = TabularCPD(
    variable="NotaExamen",
    variable_card=2,
    values=[[0.70, 0.20], [0.30, 0.80]],
    evidence=["Asistencia"],
    evidence_card=[2],
    state_names={"NotaExamen": ["Baja", "Alta"], "Asistencia": ["Baja", "Alta"]}
)

cpd_aprobacion = TabularCPD(
    variable="Aprobacion",
    variable_card=2,
    values=[[0.95, 0.10], [0.05, 0.90]],
    evidence=["NotaExamen"],
    evidence_card=[2],
    state_names={"Aprobacion": ["Desaprueba", "Aprueba"], "NotaExamen": ["Baja", "Alta"]}
)

model.add_cpds(cpd_horas, cpd_asistencia, cpd_nota, cpd_aprobacion)
model.check_model()

inferencia = VariableElimination(model)

# --- PARTE 1: MOSTRAR TODOS LOS ESCENARIOS COMBINADOS ---
print("=" * 65)
print(" TABLA COMPLETA: TODOS LOS CASOS POSIBLES DE PREDICCIÓN")
print("=" * 65)
print(f"{'Horas Estudio':<15} | {'Asistencia':<12} | {'P(Desaprueba)':<15} | {'P(Aprueba)':<12}")
print("-" * 65)

for h in ["Pocas", "Muchas"]:
    for a in ["Baja", "Alta"]:
        res = inferencia.query(
            variables=["Aprobacion"],
            evidence={"HorasEstudio": h, "Asistencia": a},
            show_progress=False
        )
        p_desaprueba = res.values[0] * 100
        p_aprueba = res.values[1] * 100
        print(f"{h:<15} | {a:<12} | {p_desaprueba:>13.2f}% | {p_aprueba:>10.2f}%")

print("=" * 65)
print("\n")

# --- PARTE 2: PREGUNTAR AL USUARIO POR CONSOLA ---
print("--- CONSULTA PERSONALIZADA ---")
print("Elige las horas de estudio:")
print("  1 = Pocas")
print("  2 = Muchas")
opc_horas = input("Ingresa 1 o 2: ").strip()
horas_val = "Muchas" if opc_horas == "2" else "Pocas"

print("\nElige el nivel de asistencia:")
print("  1 = Baja")
print("  2 = Alta")
opc_asistencia = input("Ingresa 1 o 2: ").strip()
asistencia_val = "Alta" if opc_asistencia == "2" else "Baja"

resultado_usuario = inferencia.query(
    variables=["Aprobacion"],
    evidence={"HorasEstudio": horas_val, "Asistencia": asistencia_val},
    show_progress=False
)

prob_desaprueba = resultado_usuario.values[0] * 100
prob_aprueba = resultado_usuario.values[1] * 100

print("\n" + "=" * 45)
print(f" RESULTADO PARA: Horas={horas_val}, Asistencia={asistencia_val}")
print("=" * 45)
print(f"Probabilidad de Desaprobar: {prob_desaprueba:.2f}%")
print(f"Probabilidad de Aprobar:    {prob_aprueba:.2f}%")
print("=" * 45)