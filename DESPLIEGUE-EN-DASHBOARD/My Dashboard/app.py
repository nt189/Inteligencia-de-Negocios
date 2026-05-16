import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import funpymodeling.exploratory as fp
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

# ------------------------ CONFIGURACIÓN INICIAL ------------------------
st.set_page_config(page_title="Dashboard Airbnb", layout="wide")

# ---------------------- FUNCIONES --------------------------
def load_data(name):
    df = pd.read_csv(f'Datasets/{name}/listings_{name.lower()}.csv')

    numeric_df = df.select_dtypes(include=['float', 'int'])
    text_df = df.select_dtypes(include=['object'])

    return df, numeric_df.columns, text_df.columns, numeric_df

def sideBarCommonContent(pais, df=True):
    st.sidebar.title(f"DASHBOARD {pais.upper()}")
    st.sidebar.image(f'Datasets/{pais}/bandera.png')
    st.sidebar.header("Panel de selección")

    if df == False:
        st.sidebar.subheader("Gráfico normal")
        st.sidebar.subheader("Regresión lineal")
        st.sidebar.selectbox("Variables graficadas", options=[])  
        st.sidebar.button("Agregar")
        st.sidebar.subheader("Regresión logística")

# ---------------------- PANEL LATERAL ----------------------------------
col1, col2 = st.sidebar.columns(2)

with col1:
    country = st.selectbox("País", ['Mexico', 'California', 'Barcelona', 'Ottawa'])

with col2:
    view = st.selectbox("Opciones", ["Tratamiento de datos", "Analisis univariado", "Regresión lineal y multiple", "Regresión logística"])

# ------------------------ CARGA DE DATOS -------------------------------
if country:
    sideBarCommonContent(country)

    # Cargar datos solo si es la primera vez o si cambió el país
    if 'prev_country' not in st.session_state or st.session_state.prev_country != country:
        df, numeric_cols, text_cols, numeric_df = load_data(country)
        st.session_state.df = df
        st.session_state.prev_country = country
    else:
        # Usar el DataFrame modificado de la sesión
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=['float', 'int']).columns
        text_cols = df.select_dtypes(include=['object']).columns
        numeric_df = df.select_dtypes(include=['float', 'int'])

    # Actualizar las columnas con nulos basadas en el DataFrame actual
    columnas_con_nulos = df.columns[df.isnull().any()].tolist()


# ------------------------ VISTA: Tratamiento de datos -------------------------------
if view == "Tratamiento de datos":
    # --- PANEL LATERAL DE SELECCIÓN ---
    tipo_dato = st.sidebar.selectbox(
        "Tipo de datos",
        ["General", "Números enteros", "Números flotantes", "Texto"]
    )

    # --- OPCIONES DE TRATAMIENTO DE NULOS (BOTONES) ---
    st.sidebar.subheader("Tratamiento de valores nulos")

    columnas_seleccionadas = st.sidebar.multiselect(
        "Seleciona las columnas a tratar",
        options=columnas_con_nulos,
        default=[]
    )

    # --- TÍTULO DE TRATAMIENTO ---
    st.sidebar.write("Selecciona el tratamiento a aplicar:")

    # --- TÍTULO DE TRATAMIENTO ---
    st.sidebar.write("Selecciona el tratamiento a aplicar:")


    # Organizar botones en 2x2 con un botón centrado debajo
    col1, col2 = st.sidebar.columns(2)

    # --- BOTONES PARA TRATAMIENTO DE VALORES NULOS ---
    with col1:
        bfill_button = st.button("Rellenar con siguiente valor (bfill)")
        ffill_button = st.button("Rellenar con anterior valor (ffill)")

    with col2:
        mean_button = st.button("Rellenar con media")
        median_button = st.button("Rellenar con mediana")

    # --- BOTÓN PARA ELIMINAR FILAS CON NULOS (Centrado) ---
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)  # Espacio entre las filas
    dropna_button = st.sidebar.button("Eliminar filas con nulos")

    # --- OPCIONES DE TRATAMIENTO DE VALORES ATÍPICOS ---
    st.sidebar.subheader("Tratamiento de valores atípicos")

    # Seleccionar método para tratar valores atípicos
    metodo_outliers = st.sidebar.radio(
        "Selecciona el método para tratar valores atípicos:",
        options=["Cuartiles (IQR)", "Desviación estándar"]
    )

    # Botón para aplicar tratamiento
    aplicar_outliers_button = st.sidebar.button("Aplicar tratamiento de valores atípicos")

    # --- APLICAR TRATAMIENTO DE VALORES ATÍPICOS ---
    if aplicar_outliers_button:
        cuantitativas = df.select_dtypes(include=['float', 'int'])
        cualitativas = df.select_dtypes(include=['object', 'category'])

        if metodo_outliers == "Cuartiles (IQR)":
            # Método de cuartiles (IQR)
            percentiles25 = cuantitativas.quantile(0.25)
            percentiles75 = cuantitativas.quantile(0.75)
            iqr = percentiles75 - percentiles25

            Limite_Superior_iqr = percentiles75 + 1.5 * iqr
            Limite_Inferior_iqr = percentiles25 - 1.5 * iqr

            cuantitativas = cuantitativas[
                (cuantitativas <= Limite_Superior_iqr) & (cuantitativas >= Limite_Inferior_iqr)
            ]

        elif metodo_outliers == "Desviación estándar":
            # Método de desviación estándar
            Limite_Superior = cuantitativas.mean() + 3 * cuantitativas.std()
            Limite_Inferior = cuantitativas.mean() - 3 * cuantitativas.std()

            cuantitativas = cuantitativas[
                (cuantitativas <= Limite_Superior) & (cuantitativas >= Limite_Inferior)
            ]

        # Reconstruir el DataFrame con las columnas cualitativas intactas
        df = pd.concat([cuantitativas, cualitativas], axis=1)
        st.session_state.df = df  # Guardar el DataFrame actualizado
        st.success("Tratamiento de valores atípicos aplicado.")

    st.header("Información general del Dataset")

    # --- RESUMEN DEL DATASET EN COLUMNAS ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Filas", value=df.shape[0])

    with col2:
        st.metric(label="Columnas", value=df.shape[1])

    with col3:
        st.metric(label="Datos nulos", value=df.isnull().sum().sum())


    # --- CREAR INFO GENERAL ---
    info_df = pd.DataFrame({
        'Tipo de dato': df.dtypes,
        'Valores no nulos': df.notnull().sum(),
        'Valores nulos': df.isnull().sum(),
    })

    # Agregar estadísticas solo para columnas numéricas
    if not df.select_dtypes(include=['float', 'int']).empty:
        info_df['Valor mínimo'] = df.select_dtypes(include=['float', 'int']).min()
        info_df['Media'] = df.select_dtypes(include=['float', 'int']).mean()
        info_df['Mediana'] = df.select_dtypes(include=['float', 'int']).median()
        info_df['Valor máximo'] = df.select_dtypes(include=['float', 'int']).max()

    # --- FILTRAR SEGÚN SELECCIÓN ---
    if tipo_dato == "Números enteros":
        info_df = info_df[df.dtypes == 'int64']
    elif tipo_dato == "Números flotantes":
        info_df = info_df[df.dtypes == 'float64']
    elif tipo_dato == "Texto":
        info_df = info_df[df.dtypes == 'object']
    # Si es "General" no filtramos

    # --- PREPARAR PARA MOSTRAR NOMBRE DE COLUMNA ---
    info_df = info_df.reset_index().rename(columns={'index': 'Nombre de columna'})

    # --- MOSTRAR INFORMACIÓN GENERAL ---
    st.subheader("Información general de columnas")
    st.dataframe(
        info_df,
        use_container_width=True,
        height=200
    )

    # Filtrar columnas que tienen valores nulos
    columnas_con_nulos = df.columns[df.isnull().any()].tolist()

    # --- APLICAR TRATAMIENTO DE VALORES NULOS ---
    if bfill_button:
        if columnas_seleccionadas:
            df[columnas_seleccionadas] = df[columnas_seleccionadas].bfill()
            st.session_state.df = df  # Guardar el DataFrame actualizado solo si se realiza un tratamiento
            st.success("Tratamiento: Rellenado con el siguiente valor (bfill)")
        else:
            st.warning("Por favor, selecciona las columnas que quieres tratar.")

    if ffill_button:
        if columnas_seleccionadas:
            df[columnas_seleccionadas] = df[columnas_seleccionadas].ffill()
            st.session_state.df = df  # Guardar el DataFrame actualizado solo si se realiza un tratamiento
            st.success("Tratamiento: Rellenado con el valor anterior (ffill)")
        else:
            st.warning("Por favor, selecciona las columnas que quieres tratar.")

    if mean_button:
        if columnas_seleccionadas:
            # Filtrar columnas seleccionadas que sean numéricas
            columnas_numericas = [col for col in columnas_seleccionadas if df[col].dtype in ['float64', 'int64']]
            if columnas_numericas:
                for col in columnas_numericas:
                    df[col] = df[col].fillna(df[col].mean())
                st.session_state.df = df  # Guardar el DataFrame actualizado solo si se realiza un tratamiento
                st.success(f"Tratamiento: Rellenado con la media en columnas: {', '.join(columnas_numericas)}")
            else:
                st.warning("No se seleccionaron columnas numéricas para tratar con la media.")
        else:
            st.warning("Por favor, selecciona las columnas que quieres tratar.")

    if median_button:
        if columnas_seleccionadas:
            # Filtrar columnas seleccionadas que sean numéricas
            columnas_numericas = [col for col in columnas_seleccionadas if df[col].dtype in ['float64', 'int64']]
            if columnas_numericas:
                for col in columnas_numericas:
                    df[col] = df[col].fillna(df[col].median())
                st.session_state.df = df  # Guardar el DataFrame actualizado solo si se realiza un tratamiento
                st.success(f"Tratamiento: Rellenado con la mediana en columnas: {', '.join(columnas_numericas)}")
            else:
                st.warning("No se seleccionaron columnas numéricas para tratar con la mediana.")
        else:
            st.warning("Por favor, selecciona las columnas que quieres tratar.")

    if dropna_button:
        if columnas_seleccionadas:
            df = df.dropna(subset=columnas_seleccionadas)
            st.session_state.df = df  # Guardar el DataFrame actualizado solo si se realiza un tratamiento
            st.success(f"Tratamiento: Filas con nulos eliminadas en las columnas seleccionadas")
        else:
            st.warning("Por favor, selecciona las columnas que quieres tratar.")

    # --- ACTUALIZAR DATASET ---
    st.subheader("Vista completa de datos")
    st.dataframe(
        df,
        use_container_width=True,
        height=550
    )

# ------------------------ VISTA: Analisis univariado -------------------------------
elif view == "Analisis univariado":
    st.subheader("Análisis Univariado")

    # Selectbox para seleccionar una columna con una opción inicial
    columna = st.sidebar.selectbox(
        "Seleccione la columna a analizar:", 
        options=["Seleccione una columna"] + list(df.columns),
        index=0
    )

    # Checkbox para decidir si se crean intervalos
    mkintervalos = st.sidebar.checkbox("¿Crear intervalos?", value=False)

    # Verificar que se haya seleccionado una columna válida
    if columna != "Seleccione una columna":
        if columna in numeric_cols:
            try:
                # Convertir la columna a tipo numérico si es necesario
                col_categorizada = df[columna].replace('%', '', regex=True).astype(float)

                # Obtener estadísticas básicas
                n = df[columna].shape[0]
                Max = df[columna].max()
                Min = df[columna].min()
                R = Max - Min  # Rango

                # Calcular el número de intervalos usando la regla de Sturges
                ni = 1 + 3.32 * np.log10(n)
                ni = round(ni)

                # Calcular el ancho del intervalo
                i = R / ni

                # Crear los intervalos
                intervalos = np.linspace(Min, Max, ni + 1)

                # Crear categorías basadas en los intervalos
                if mkintervalos:
                    categorias = [f"{intervalos[j]:.2f} - {intervalos[j+1]:.2f}" for j in range(ni)]
                else:
                    categorias = [f"{intervalos[j]:.2f}" for j in range(ni)]

                # Asignar categorías a la columna
                col_categorizada = pd.cut(df[columna], bins=intervalos, labels=categorias, right=False)

                # Crear la tabla de frecuencias
                table1 = fp.freq_tbl(col_categorizada)
                table2 = table1.drop(['percentage', 'cumulative_perc'], axis=1)

                # Filtrar frecuencias mayores a un valor mínimo
                x = st.sidebar.number_input("Frecuencia mínima", min_value=1, max_value=table2['frequency'].max(), value=1)
                Filtro = table2[table2['frequency'] > x]

                # Verificar si hay datos después del filtrado
                if Filtro.empty:
                    st.warning("No hay datos suficientes después del filtrado. Ajuste los parámetros.")
                else:
                    # Crear el índice de la tabla
                    Filtro_index = Filtro.set_index(columna)
            except Exception as e:
                st.error(f"Error al procesar la columna seleccionada: {e}")
        elif columna in text_cols:
            try:
                # Crear la tabla de frecuencias para columnas de texto
                table1 = fp.freq_tbl(df[columna])
                table2 = table1.drop(['percentage', 'cumulative_perc'], axis=1)

                # Filtrar frecuencias mayores a un valor mínimo
                x = st.sidebar.number_input("Frecuencia mínima", min_value=1, max_value=table2['frequency'].max(), value=1)
                Filtro = table2[table2['frequency'] > x]

                # Verificar si hay datos después del filtrado
                if Filtro.empty:
                    st.warning("No hay datos suficientes después del filtrado. Ajuste los parámetros.")
                else:
                    # Crear el índice de la tabla
                    Filtro_index = Filtro.set_index(columna)
            except Exception as e:
                st.error(f"Error al procesar la columna seleccionada: {e}")
        else:
            st.warning("La columna seleccionada no es válida para este análisis.")
    else:
        st.info("Por favor, seleccione una columna para continuar.")

    # Botones para gráficos
    lineplot_btn = st.sidebar.button("Gráfico de líneas")
    scatterplot_btn = st.sidebar.button("Gráfico de dispersión")
    barplot_btn = st.sidebar.button("Gráfico de barras")
    pieplot_btn = st.sidebar.button("Gráfico de pastel")


    # Gráfico de líneas
    if lineplot_btn:
        if 'Filtro' in locals() and not Filtro.empty and 'frequency' in Filtro.columns:
            fig = px.line(Filtro_index, x=Filtro_index.index, y='frequency', title=f"Gráfico de líneas de {columna}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para generar el gráfico de líneas.")

    # Gráfico de dispersión
    if scatterplot_btn:
        if 'Filtro' in locals() and not Filtro.empty and 'frequency' in Filtro.columns:
            fig = px.scatter(Filtro_index, x=Filtro_index.index, y='frequency', title=f"Gráfico de dispersión de {columna}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para generar el gráfico de dispersión.")

    # Gráfico de barras
    if barplot_btn:
        if 'Filtro' in locals() and not Filtro.empty and 'frequency' in Filtro.columns:
            fig = px.bar(Filtro_index, x=Filtro_index.index, y='frequency', title=f"Gráfico de barras de {columna}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para generar el gráfico de barras.")

    # Gráfico de pastel
    if pieplot_btn:
        if 'Filtro' in locals() and not Filtro.empty and 'frequency' in Filtro.columns:
            fig = px.pie(Filtro_index, names=Filtro_index.index, values='frequency', title=f"Gráfico de pastel de {columna}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para generar el gráfico de pastel.")


    # Resumen de la tabla
    if 'Filtro' in locals() and not Filtro_index.empty:
        st.subheader("Resumen de la tabla")
        st.dataframe(Filtro_index)
    else:
        st.warning("No hay datos suficientes para mostrar el resumen.")

# ------------------------ VISTA: Regrecion lineal y multiple -------------------------------
elif view == "Regresión lineal y multiple":    
    st.subheader("Regresiones lineales")

    # Conversión de variables categóricas a numéricas
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        mapping = dict(enumerate(df[col].astype("category").cat.categories))
        print(f"Mapping for {col}: {mapping}")
        df[col] = df[col].astype("category").cat.codes


    dep = st.sidebar.selectbox("Variable dependiente", options=df.columns, key="dependent_var")
    indep = st.sidebar.multiselect("Variables independientes", options=df.columns, key="independent_vars")

    graficar = st.sidebar.button("Graficar")

    if graficar:
        var_dep = df[dep]
        vars_indep = df[indep]

        model = LinearRegression()
        model.fit(X=vars_indep, y=var_dep)
        y_pred = model.predict(vars_indep)

        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Coeficiente de determinación R2", value=str(model.score(vars_indep, var_dep)))
        with col2:
            st.metric(label="Coeficiente de correlación", value=str(np.sqrt(model.score(vars_indep, var_dep))))
        with col3:
            st.metric(label="Modelo matemático", value='y = ' + str(model.coef_) + ' | X = ' + str(model.intercept_))

        # Gráficos
        if len(indep) == 1:
            df_plot = pd.DataFrame({
                'X': vars_indep[indep[0]],
                'Real': var_dep,
                'Predicción': y_pred
            })

            fig = go.Figure()
            fig.add_scatter(x=df_plot['X'], y=df_plot['Real'], mode='markers', name='Real')
            fig.add_scatter(x=df_plot['X'], y=df_plot['Predicción'], mode='markers', name='Predicción', marker=dict(color='red'))
            st.plotly_chart(fig)

            # Mostrar tabla de valores reales y predicciones
            st.subheader("Tabla de valores reales y predicciones")
            st.dataframe(df_plot)

        elif len(indep) > 1:
            # Gráfico de Real vs Predicción
            df_resultado = pd.DataFrame({
                'Real': var_dep,
                'Predicción': y_pred
            })

            fig = px.scatter(df_resultado, x='Real', y='Predicción', title='Real vs Predicción')

            # Puntos de referencia: Real == Predicción (línea perfecta como puntos rojos)
            df_referencia = pd.DataFrame({'Real': df_resultado['Real'], 'Predicción': df_resultado['Real']})
            fig.add_scatter(x=df_referencia['Real'], y=df_referencia['Predicción'],
                            mode='markers', name='Referencia perfecta', marker=dict(color='red', size=4))

            st.plotly_chart(fig)

            # Tabla comparativa
            st.dataframe(df_resultado)

    # Gráfico de correlaciones
    corr_matrix = df.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, title="Matriz de correlaciones")
    st.plotly_chart(fig_corr)

# ------------------------ VISTA: Regresión logística -------------------------------
elif view == "Regresión logística":    
    st.subheader("Regresiones lineales y logísticas")

    # Conversión de variables categóricas a numéricas
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        mapping = dict(enumerate(df[col].astype("category").cat.categories))
        print(f"Mapping for {col}: {mapping}")
        df[col] = df[col].astype("category").cat.codes

    dep = st.sidebar.selectbox("Variable dependiente", options=df.columns, key="dependent_var")
    indep = st.sidebar.multiselect("Variables independientes", options=df.columns, key="independent_vars")

    # Convertir la variable dependiente en dicotómica
    if dep:
        threshold = st.sidebar.slider("Umbral para dicotomizar la variable dependiente", 
                                    min_value=float(df[dep].min()), 
                                    max_value=float(df[dep].max()), 
                                    value=float(df[dep].median()))
        df[dep] = (df[dep] > threshold).astype(int)

        # Validar que haya al menos dos clases después de la dicotomización
        unique_classes = df[dep].unique()
        if len(unique_classes) < 2:
            st.warning("El umbral seleccionado no divide los datos en dos clases. Todos los valores son iguales. Por favor, ajusta el umbral.")
    else:
        class_counts = df[dep].value_counts()
        if class_counts.min() < 5:  # Advertencia si una clase tiene muy pocos datos
            st.warning(f"Una de las clases tiene muy pocos datos ({class_counts.min()} muestras). Esto puede afectar el modelo.")

    tstz = st.sidebar.slider("Tamaño del conjunto de prueba", min_value=0.1, max_value=0.5, value=0.3)

    pslbl = st.sidebar.selectbox("Etiqueta positiva", options=unique_classes, index=0)

    # Mostrar el botón para continuar
    Mostrar_datos = st.sidebar.button("Mostrar datos")

    if Mostrar_datos:
        # Continuar con el resto del código...
        X = df[indep]
        y = df[dep]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tstz, random_state=None)

        # Escalar los datos
        escalar = StandardScaler()
        X_train = escalar.fit_transform(X_train)
        X_test = escalar.transform(X_test)

        # Entrenar el modelo
        algoritmo = LogisticRegression()
        algoritmo.fit(X_train, y_train)

        # Realizar predicciones
        y_pred = algoritmo.predict(X_test)

        # Calcular métricas y mostrar resultados
        matriz = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary", pos_label=pslbl)
        exactitud = accuracy_score(y_test, y_pred)
        sensibilidad = recall_score(y_test, y_pred, average="binary", pos_label=pslbl)

        tbl_info = pd.DataFrame({
            'Variable Dependiente': [dep],
            'Variables Independientes': [", ".join(indep)],
            'Precisión': [precision],
            'Exactitud': [exactitud],
            'Sensibilidad': [sensibilidad],
            'Matriz de Confusión': [matriz.tolist()]
        })

        st.write("Resultados del modelo:")
        st.dataframe(tbl_info)

        st.write("Matriz de Confusión:")
        fig = px.imshow(matriz, text_auto=True, labels=dict(x="Predicción", y="Real"))
        st.plotly_chart(fig)

