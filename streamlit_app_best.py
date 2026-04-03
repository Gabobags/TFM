
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Airbnb Pricing Intelligence NYC",
    page_icon="🏙️",
    layout="wide"
)

DEFAULT_2019_PATH = "AB_NYC_2019.csv"
DEFAULT_2024_PATH = "AB_NYC_2024.csv"

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def prepare_dataset(df: pd.DataFrame):
    data = df.copy()

    if "reviews_per_month" in data.columns:
        data["reviews_per_month"] = data["reviews_per_month"].fillna(0)

    if "number_of_reviews" in data.columns:
        data["has_reviews"] = (data["number_of_reviews"] > 0).astype(int)
    else:
        data["has_reviews"] = 0

    if "availability_365" in data.columns:
        data["is_available"] = (data["availability_365"] > 0).astype(int)
    else:
        data["is_available"] = 0

    cols_drop = [c for c in ["id", "name", "host_id", "host_name", "last_review"] if c in data.columns]
    data = data.drop(columns=cols_drop)

    required_cols = [
        "neighbourhood_group", "neighbourhood", "latitude", "longitude",
        "room_type", "price", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count",
        "availability_365", "has_reviews", "is_available"
    ]

    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    data = data[required_cols].copy()

    q01 = data["price"].quantile(0.01)
    q99 = data["price"].quantile(0.99)

    data = data[(data["price"] >= q01) & (data["price"] <= q99)].copy()
    data["log_price"] = np.log1p(data["price"])

    metrics = {
        "rows_used": int(len(data)),
        "price_p1": float(q01),
        "price_p99": float(q99),
        "median_price": float(data["price"].median()),
        "mean_price": float(data["price"].mean())
    }
    return data, metrics

@st.cache_resource
def train_model(df: pd.DataFrame, year_label: str):
    data, metrics = prepare_dataset(df)

    X = data.drop(columns=["price", "log_price"])
    y = data["log_price"]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    r2_test = float(pipeline.score(X_test, y_test))

    model_info = {
        "year": year_label,
        "pipeline": pipeline,
        "data_clean": data,
        "metrics": {**metrics, "r2_test": r2_test}
    }
    return model_info

def build_listing_input(source_df: pd.DataFrame):
    st.subheader("Características del alojamiento")

    col1, col2, col3 = st.columns(3)

    district = col1.selectbox(
        "Distrito",
        sorted(source_df["neighbourhood_group"].dropna().unique().tolist())
    )

    neighbourhood_options = sorted(
        source_df.loc[source_df["neighbourhood_group"] == district, "neighbourhood"]
        .dropna()
        .unique()
        .tolist()
    )
    neighbourhood = col1.selectbox("Barrio", neighbourhood_options)

    room_type = col2.selectbox(
        "Tipo de alojamiento",
        sorted(source_df["room_type"].dropna().unique().tolist())
    )

    minimum_nights = col2.slider("Noches mínimas", 1, 365, 2)
    availability_365 = col2.slider("Disponibilidad anual", 0, 365, 180)

    number_of_reviews = col3.slider("Número de reseñas", 0, 500, 10)
    reviews_per_month = col3.slider("Reseñas por mes", 0.0, 20.0, 0.5, 0.1)
    host_listings_count = col3.slider("Anuncios del anfitrión", 1, 50, 1)

    location_match = source_df[
        (source_df["neighbourhood_group"] == district) &
        (source_df["neighbourhood"] == neighbourhood)
    ]

    if not location_match.empty:
        latitude = float(location_match["latitude"].median())
        longitude = float(location_match["longitude"].median())
    else:
        latitude = float(source_df["latitude"].median())
        longitude = float(source_df["longitude"].median())

    listing = pd.DataFrame([{
        "neighbourhood_group": district,
        "neighbourhood": neighbourhood,
        "latitude": latitude,
        "longitude": longitude,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": host_listings_count,
        "availability_365": availability_365,
        "has_reviews": int(number_of_reviews > 0),
        "is_available": int(availability_365 > 0)
    }])

    return listing, {
        "district": district,
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "availability_365": availability_365,
        "host_listings_count": host_listings_count
    }

def predict_price(model_info, listing):
    pred_log = model_info["pipeline"].predict(listing)[0]
    pred_price = float(np.expm1(pred_log))
    return pred_price

def comparable_market_stats(data_clean: pd.DataFrame, district: str, room_type: str):
    subset = data_clean[
        (data_clean["neighbourhood_group"] == district) &
        (data_clean["room_type"] == room_type)
    ]
    if subset.empty:
        subset = data_clean

    return {
        "count": int(len(subset)),
        "median_price": float(subset["price"].median()),
        "mean_price": float(subset["price"].mean()),
        "p25": float(subset["price"].quantile(0.25)),
        "p75": float(subset["price"].quantile(0.75))
    }

def comparison_message(price_2019: float, price_2024: float):
    diff = price_2024 - price_2019
    pct = (diff / price_2019 * 100) if price_2019 else 0
    if diff > 0:
        return f"El modelo 2024 sugiere un precio **{abs(diff):.2f} USD** más alto que el modelo 2019 (**+{pct:.1f}%**)."
    elif diff < 0:
        return f"El modelo 2024 sugiere un precio **{abs(diff):.2f} USD** más bajo que el modelo 2019 (**{pct:.1f}%**)."
    return "Ambos modelos sugieren prácticamente el mismo precio."

# -----------------------------
# UI
# -----------------------------
st.title("🏙️ Airbnb Pricing Intelligence NYC")
st.caption(
    "Demo de productivización del proyecto: recomendación de precios con perspectiva temporal "
    "basada en los datasets de 2019 y 2024."
)

with st.sidebar:
    st.header("Configuración")
    path_2019 = st.text_input("Ruta CSV 2019", value=DEFAULT_2019_PATH)
    path_2024 = st.text_input("Ruta CSV 2024", value=DEFAULT_2024_PATH)

    mode = st.radio(
        "Modo de estimación",
        [
            "Estimación actual basada en 2024",
            "Comparar 2019 vs 2024",
            "Usar solo modelo 2019",
            "Usar solo modelo 2024"
        ]
    )

    st.markdown("---")
    st.write(
        "La opción **Estimación actual basada en 2024** no representa un precio en tiempo real, "
        "sino una aproximación basada en el dataset más reciente disponible."
    )

# Load + train
try:
    df_2019 = load_data(path_2019)
    df_2024 = load_data(path_2024)

    model_2019 = train_model(df_2019, "2019")
    model_2024 = train_model(df_2024, "2024")
except Exception as e:
    st.error(f"No se pudieron cargar o entrenar los modelos: {e}")
    st.stop()

# Summary metrics
st.subheader("Resumen de datasets y modelos")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Filas limpias 2019", f"{model_2019['metrics']['rows_used']:,}")
m2.metric("R² modelo 2019", f"{model_2019['metrics']['r2_test']:.3f}")
m3.metric("Filas limpias 2024", f"{model_2024['metrics']['rows_used']:,}")
m4.metric("R² modelo 2024", f"{model_2024['metrics']['r2_test']:.3f}")

with st.expander("Ver métricas de limpieza"):
    info_df = pd.DataFrame([
        {
            "Dataset": "2019",
            "Percentil 1 price": model_2019["metrics"]["price_p1"],
            "Percentil 99 price": model_2019["metrics"]["price_p99"],
            "Media price": round(model_2019["metrics"]["mean_price"], 2),
            "Mediana price": round(model_2019["metrics"]["median_price"], 2),
        },
        {
            "Dataset": "2024",
            "Percentil 1 price": model_2024["metrics"]["price_p1"],
            "Percentil 99 price": model_2024["metrics"]["price_p99"],
            "Media price": round(model_2024["metrics"]["mean_price"], 2),
            "Mediana price": round(model_2024["metrics"]["median_price"], 2),
        }
    ])
    st.dataframe(info_df, use_container_width=True)

# Listing form
source_df_for_inputs = df_2024 if "2024" in mode else df_2019
listing, user_choices = build_listing_input(source_df_for_inputs)

calculate = st.button("Calcular precio recomendado", type="primary", use_container_width=True)

if calculate:
    price_2019 = predict_price(model_2019, listing)
    price_2024 = predict_price(model_2024, listing)

    market_2019 = comparable_market_stats(
        model_2019["data_clean"], user_choices["district"], user_choices["room_type"]
    )
    market_2024 = comparable_market_stats(
        model_2024["data_clean"], user_choices["district"], user_choices["room_type"]
    )

    st.subheader("Resultado de la estimación")

    if mode == "Estimación actual basada en 2024":
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio estimado actual (basado en 2024)", f"${price_2024:,.2f}")
        c2.metric("Mediana del segmento comparable 2024", f"${market_2024['median_price']:,.2f}")
        c3.metric("Rango intercuartílico 2024", f"${market_2024['p25']:,.0f} - ${market_2024['p75']:,.0f}")

        st.info(
            "Esta estimación representa una aproximación actual basada en el dataset más reciente disponible (2024), "
            "por lo que sirve como referencia actualizada de mercado, aunque no constituye una predicción en tiempo real."
        )

    elif mode == "Comparar 2019 vs 2024":
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Precio estimado según mercado 2019", f"${price_2019:,.2f}")
            st.metric("Mediana comparable 2019", f"${market_2019['median_price']:,.2f}")
        with c2:
            st.metric("Precio estimado según mercado 2024", f"${price_2024:,.2f}")
            st.metric("Mediana comparable 2024", f"${market_2024['median_price']:,.2f}")

        st.success(comparison_message(price_2019, price_2024))

        diff_market = market_2024["median_price"] - market_2019["median_price"]
        st.write(
            f"En el segmento comparable seleccionado, la mediana de mercado pasa de "
            f"**${market_2019['median_price']:,.2f}** en 2019 a **${market_2024['median_price']:,.2f}** en 2024."
        )

    elif mode == "Usar solo modelo 2019":
        c1, c2 = st.columns(2)
        c1.metric("Precio estimado según mercado 2019", f"${price_2019:,.2f}")
        c2.metric("Mediana del segmento comparable 2019", f"${market_2019['median_price']:,.2f}")

    elif mode == "Usar solo modelo 2024":
        c1, c2 = st.columns(2)
        c1.metric("Precio estimado según mercado 2024", f"${price_2024:,.2f}")
        c2.metric("Mediana del segmento comparable 2024", f"${market_2024['median_price']:,.2f}")

    st.subheader("Detalles del alojamiento evaluado")
    st.dataframe(listing, use_container_width=True)

    st.subheader("Interpretación automática")
    bullets = []

    if price_2024 > price_2019:
        bullets.append("El mercado reciente sugiere una presión alcista sobre el precio.")
    elif price_2024 < price_2019:
        bullets.append("El mercado reciente sugiere una reducción relativa del precio esperado.")
    else:
        bullets.append("La lógica de pricing se mantiene estable entre 2019 y 2024 para este perfil.")

    if user_choices["number_of_reviews"] == 0:
        bullets.append("La ausencia de reseñas puede afectar la confianza del mercado y moderar el precio estimado.")

    if user_choices["availability_365"] >= 180:
        bullets.append("Una alta disponibilidad puede actuar como señal de gestión más profesional del alojamiento.")

    if user_choices["room_type"] == "Entire home/apt":
        bullets.append("Los alojamientos completos suelen ubicarse entre los segmentos de mayor valor del mercado.")

    for b in bullets:
        st.write(f"- {b}")

st.markdown("---")
st.subheader("Aplicación práctica y valor de negocio")
st.write(
    "Esta aplicación demuestra cómo el modelo puede productivizarse en un entorno empresarial. "
    "Un anfitrión, gestor de propiedades o analista puede introducir las características de un alojamiento "
    "y obtener una recomendación de precio, así como una comparación entre la lógica del mercado de 2019 y 2024. "
    "De esta forma, la herramienta no solo actúa como motor de predicción, sino también como sistema de apoyo a la decisión con perspectiva temporal."
)
