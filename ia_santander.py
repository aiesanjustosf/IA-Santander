# ia_santander.py
# Herramienta para uso interno - AIE San Justo
# App EXCLUSIVA para Banco Santander (Cuenta Corriente)

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --- UI / assets ---
HERE = Path(__file__).parent
LOGO = HERE / "logo_aie.png"
FAVICON = HERE / "favicon-aie.ico"

st.set_page_config(
    page_title="IA Resumen Santander - AIE San Justo",
    page_icon=str(FAVICON) if FAVICON.exists() else None,
    layout="wide",
)

if LOGO.exists():
    st.image(str(LOGO), width=200)

st.title("IA Resumen Santander")
st.caption("Asesoramiento Integral de Empresas San Justo - Uso interno AIE")


# --- deps diferidas ---
try:
    import pdfplumber
except Exception as e:
    st.error(
        "No se pudo importar pdfplumber.\n\n"
        f"Detalle del error: {e}\n"
        "Revisá `requirements.txt` e instalá la dependencia."
    )
    st.stop()


# --- helpers numéricos ---

def _parse_amount(text: str) -> float:
    """
    Convierte importe Santander '5.000.000,00' → 5000000.00 (float).
    Si viene vacío, devuelve 0.0
    """
    if text is None:
        return 0.0
    s = str(text).strip()
    if not s or s == "-":
        return 0.0
    # limpiamos símbolos
    s = s.replace("$", "").replace("\u00a0", " ").replace(" ", "")
    # puntos miles y coma decimal
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


def _clasificar_movimiento(desc: str) -> str:
    """
    Clasificación simple por descripción, para resumen impositivo.
    Ajustable según necesidad.
    """
    d = (desc or "").lower()

    if "impuesto ley 25.413" in d or "impuesto ley 25413" in d:
        if "debito" in d:
            return "IMP. LEY 25.413 - DÉBITOS"
        if "credito" in d:
            return "IMP. LEY 25.413 - CRÉDITOS"
        return "IMP. LEY 25.413"

    if "sircreb" in d:
        return "SIRCREB"

    if "iva 21" in d or "iva 21% reg" in d:
        return "IVA 21%"

    if "iva percepcion" in d or "iva percepción" in d:
        return "IVA PERCEPCIÓN"

    if "comision por servicio de cuenta" in d:
        return "COMISIÓN CUENTA"

    if "pago haberes" in d:
        return "PAGO HABERES"

    if "deposito de efectivo" in d or "deposito efvo" in d:
        return "DEPÓSITO EFECTIVO"

    return "OTROS"


# --- parser específico Santander ---

def parse_santander_pdf(file_bytes: bytes) -> pd.DataFrame:
    """
    Lee un PDF de Resumen de Cuenta Banco Santander (Cuenta Corriente)
    y devuelve un DataFrame con los movimientos.

    Columnas:
      - fecha (datetime)
      - comprobante
      - movimiento
      - debito
      - credito
      - saldo
      - tipo  (DEBITO / CREDITO / '')
      - importe        (signado: + créditos, - débitos)
      - importe_raw    (siempre positivo)
      - categoria      (clasificación impositiva básica)
    """
    rows = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                if not table or len(table) < 2:
                    continue

                # Primera fila: cabecera
                header = [(c or "").strip().lower() for c in table[0]]
                header_join = " ".join(header)

                # Buscamos la tabla principal de movimientos
                if "fecha" in header_join and "saldo" in header_join:
                    # Esperamos algo como: Fecha | Comprobante | Movimiento | Débito | Crédito | Saldo ...
                    for raw_row in table[1:]:
                        if not raw_row or not any(raw_row):
                            continue

                        cells = [(c or "").strip() for c in raw_row]
                        # aseguramos al menos 6 columnas
                        if len(cells) < 6:
                            cells += [""] * (6 - len(cells))

                        fecha_txt, comp, mov, deb_txt, cred_txt, saldo_txt = cells[:6]

                        # Filtramos filas que NO son movimientos (totales, textos, etc.)
                        if not re.match(r"\d{2}/\d{2}/\d{2}", fecha_txt):
                            continue

                        deb = _parse_amount(deb_txt)
                        cred = _parse_amount(cred_txt)
                        saldo = _parse_amount(saldo_txt)

                        # Determinamos tipo e importe signado
                        if abs(deb) > 0:
                            tipo = "DEBITO"
                            importe_raw = deb
                            importe = -deb
                        elif abs(cred) > 0:
                            tipo = "CREDITO"
                            importe_raw = cred
                            importe = cred
                        else:
                            tipo = ""
                            importe_raw = 0.0
                            importe = 0.0

                        rows.append(
                            {
                                "fecha_str": fecha_txt,
                                "comprobante": comp,
                                "movimiento": mov,
                                "debito": deb,
                                "credito": cred,
                                "saldo": saldo,
                                "tipo": tipo,
                                "importe": importe,
                                "importe_raw": importe_raw,
                            }
                        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Parseo de fecha
    df["fecha"] = pd.to_datetime(
        df["fecha_str"], format="%d/%m/%y", errors="coerce"
    )

    # Clasificación básica
    df["categoria"] = df["movimiento"].apply(_clasificar_movimiento)

    # Orden estándar
    df = df[
        [
            "fecha",
            "fecha_str",
            "comprobante",
            "movimiento",
            "categoria",
            "tipo",
            "debito",
            "credito",
            "importe",
            "importe_raw",
            "saldo",
        ]
    ].sort_values(["fecha", "comprobante"], ignore_index=True)

    return df


def construir_resumen_impositivo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arma un resumen impositivo agrupado por categoría impositiva.
    """
    if df.empty:
        return pd.DataFrame()

    # Sólo categorías que nos interesan para impuestos
    categorias_interes = [
        "IMP. LEY 25.413 - DÉBITOS",
        "IMP. LEY 25.413 - CRÉDITOS",
        "IMP. LEY 25.413",
        "SIRCREB",
        "IVA 21%",
        "IVA PERCEPCIÓN",
        "COMISIÓN CUENTA",
    ]

    df_imp = df[df["categoria"].isin(categorias_interes)].copy()
    if df_imp.empty:
        return pd.DataFrame()

    resumen = (
        df_imp.groupby("categoria", as_index=False)["importe_raw"]
        .sum()
        .rename(columns={"categoria": "Concepto", "importe_raw": "Importe"})
    )

    # Orden razonable
    orden = {c: i for i, c in enumerate(categorias_interes)}
    resumen["__orden"] = resumen["Concepto"].map(orden).fillna(999)
    resumen = resumen.sort_values("__orden").drop(columns="__orden").reset_index(
        drop=True
    )

    return resumen


def calcular_saldos(df: pd.DataFrame) -> tuple[float, float]:
    """
    Determina saldo inicial y saldo final.
    - saldo inicial: fila cuyo movimiento contiene 'Saldo Inicial' (si existe)
      si no, se toma el primer saldo disponible.
    - saldo final: último saldo
    """
    if df.empty:
        return 0.0, 0.0

    saldo_inicial = None

    # Buscamos explícitamente 'Saldo Inicial' en descripción
    mask_si = df["movimiento"].str.contains("saldo inicial", case=False, na=False)
    if mask_si.any():
        saldo_inicial = df.loc[mask_si, "saldo"].iloc[0]
    else:
        # fallback: primer saldo
        saldo_inicial = df["saldo"].iloc[0]

    saldo_final = df["saldo"].iloc[-1]

    return float(saldo_inicial), float(saldo_final)


# ============================
#            UI
# ============================

st.markdown("### 1️⃣ Cargar resumen de cuenta Santander (PDF)")

uploaded = st.file_uploader(
    "Subí el PDF del **Resumen de Cuenta Corriente Banco Santander**",
    type=["pdf"],
)

if not uploaded:
    st.info("Esperando que subas un PDF de Santander…")
    st.stop()

# Parseamos PDF
file_bytes = uploaded.read()
df_mov = parse_santander_pdf(file_bytes)

if df_mov.empty:
    st.error(
        "No se pudieron detectar movimientos Santander en el PDF.\n\n"
        "Verificá que el resumen sea de **Cuenta Corriente en pesos** "
        "y tenga las columnas Fecha / Comprobante / Movimiento / Débito / Crédito / Saldo."
    )
    st.stop()

st.success(f"Se detectaron {len(df_mov)} movimientos de Banco Santander.")

# --- Resumen de saldos y totales ---

saldo_inicial, saldo_final = calcular_saldos(df_mov)
total_debitos = df_mov.loc[df_mov["tipo"] == "DEBITO", "importe_raw"].sum()
total_creditos = df_mov.loc[df_mov["tipo"] == "CREDITO", "importe_raw"].sum()

st.markdown("### 2️⃣ Resumen general de movimientos")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Saldo inicial", f"$ {saldo_inicial:,.2f}")
with c2:
    st.metric("Saldo final", f"$ {saldo_final:,.2f}")
with c3:
    st.metric("Total débitos", f"$ {total_debitos:,.2f}")
with c4:
    st.metric("Total créditos", f"$ {total_creditos:,.2f}")

st.markdown("### 3️⃣ Detalle de movimientos")

st.dataframe(
    df_mov[
        [
            "fecha",
            "comprobante",
            "movimiento",
            "categoria",
            "tipo",
            "debito",
            "credito",
            "saldo",
        ]
    ],
    use_container_width=True,
    height=420,
)

# --- Resumen impositivo ---

st.markdown("### 4️⃣ Resumen impositivo (según movimientos detectados)")

df_imp = construir_resumen_impositivo(df_mov)

if df_imp.empty:
    st.info(
        "No se detectaron conceptos impositivos (Ley 25.413, SIRCREB, IVA, etc.) "
        "en los movimientos analizados."
    )
else:
    st.dataframe(df_imp, use_container_width=True)

# --- Exportar a Excel (NO CSV) ---

st.markdown("### 5️⃣ Exportar a Excel")

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
    df_mov.to_excel(writer, index=False, sheet_name="Movimientos")
    if not df_imp.empty:
        df_imp.to_excel(writer, index=False, sheet_name="Resumen_impositivo")

st.download_button(
    "⬇️ Descargar Excel Santander",
    data=buffer.getvalue(),
    file_name="santander_resumen_aie.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    help="Exporta los movimientos y el resumen impositivo a un archivo Excel.",
)

st.caption(
    "⚠️ Este análisis es de uso interno AIE. Verificá siempre los importes "
    "contra el resumen original del Banco Santander."
)
