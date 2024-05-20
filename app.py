import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import hashlib
import ast
import folium
from streamlit_folium import folium_static, st_folium
import requests
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import os

# Imprime el valor de las variables de entorno MAILGUN_DOMAIN y MAILGUN_API_KEY
st.write("Valor de MAILGUN_DOMAIN:", os.getenv('MAILGUN_DOMAIN'))
st.write("Valor de MAILGUN_API_KEY:", os.getenv('MAILGUN_API_KEY'))

# Funtions for user authentication

def hash_password(password):
    """
    Hashes the given password using SHA-256 algorithm.

    Parameters:
    password (str): The password to be hashed.

    Returns:
    str: The hashed password as a hexadecimal string.
    """
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """
    Load users from a CSV file if it exists, otherwise return an empty DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded users or an empty DataFrame if the file doesn't exist.
    """
    if Path('users.csv').exists():
        return pd.read_csv('users.csv')
    else:
        return pd.DataFrame(columns=['username', 'password'])

# Funcion para ver la informacion del desarrollador
def show_developer_info():
    st.subheader('Información del Desarrollador')
    st.write("""
    Esta aplicación ha sido desarrollada por Kevin Andrés Blandón Vélez, estudiante de la Universidad Nacional de Colombia - Sede Medellín.
    Para más información, puedes contactarme a través de mi correo electrónico:
    - Email: kblandonv@unal.edu.co
    - GitHub: [kblandonv](https://github.com/kblandonv)
    """)

# Función para guardar un nuevo usuario
def save_user(username, password, email, role):
    """
    Saves a new user to the users.csv file.

    Parameters:
    - username (str): The username of the new user.
    - password (str): The password of the new user.
    - email (str): The email address of the new user.
    - role (str): The role of the new user.

    Returns:
    None
    """
    hashed_password = hash_password(password)
    new_user = pd.DataFrame({
        'username': [username],
        'password': [hashed_password],
        'email': [email],
        'role': [role]
    })
    file_path = 'data/users.csv'
    new_user.to_csv(file_path, mode='a', header=not Path(file_path).exists(), index=False)

# Función para autenticar usuarios
def authenticate_user(username, password):
    """
    Authenticates a user by checking if the provided username and password match any user in the users.csv file.

    Parameters:
    - username (str): The username of the user.
    - password (str): The password of the user.

    Returns:
    - dict or None: A dictionary containing the user information if the authentication is successful, or None if the authentication fails.
    """
    hashed_password = hash_password(password)
    file_path = 'data/users.csv'
    if Path(file_path).exists():
        users = pd.read_csv(file_path)
        user = users[(users['username'] == username) & (users['password'] == hashed_password)]
        if not user.empty:
            return user.iloc[0].to_dict()
    return None

# Función para verificar si un usuario es administrador
def is_admin(role):
    return role == 'admin'


#Funcion para actualizar contraseña de usuario
def update_password(username, password):
    """
    Updates the password of a user in the users.csv file.

    Parameters:
    - username (str): The username of the user.
    - password (str): The new password for the user.

    Returns:
    None
    """
    hashed_password = hash_password(password)
    file_path = 'data/users.csv'
    if Path(file_path).exists():
        users = pd.read_csv(file_path)
        user_index = users[users['username'] == username].index
        users.loc[user_index, 'password'] = hashed_password
        users.to_csv(file_path, index=False)

# Función para enviar email usando Mailgun
def send_email(subject, message, description, recipients):
    """
    Envía un correo electrónico usando la API de Mailgun.

    Args:
        subject (str): El asunto del correo electrónico.
        message (str): El contenido principal del correo electrónico.
        description (str): Descripción adicional que se incluirá en el correo electrónico.
        recipients (list): Lista de direcciones de correo electrónico de los destinatarios.

    Returns:
        requests.Response: El objeto de respuesta devuelto por la API de Mailgun.
    """
    mailgun_domain = os.getenv('MAILGUN_DOMAIN')
    mailgun_api_key = os.getenv('MAILGUN_API_KEY')

    if not mailgun_domain or not mailgun_api_key:
        raise ValueError("Las variables de entorno MAILGUN_DOMAIN y MAILGUN_API_KEY no están configuradas correctamente.")

    import requests

    return requests.post(
        f"https://api.mailgun.net/v3/{mailgun_domain}/messages",
        auth=("api", mailgun_api_key),
        data={
            "from": f"Excited User <mailgun@{mailgun_domain}>",
            "to": recipients,
            "subject": subject,
            "text": f"{message}\n\n{description}"
        }
    )


# Cargar datos de avenidas torrenciales
avenidas_torrenciales_data = pd.read_csv('data/avenidas_torrenciales.csv')

# Función para entrenar y evaluar un modelo SVM
def train_and_evaluate_svm(data):
    """
    Trains and evaluates a Support Vector Machine (SVM) model using the given data.

    Parameters:
    - data: A pandas DataFrame containing the input features and target variable.

    Returns:
    None
    """
    # Preprocesamiento de datos
    X = data.drop(columns=['codigo', 'nombre', 'tipo_amenaza', 'categoria_proteccion', 'subcategoria_proteccion', 'Shape', 'Shape_Length', 'Shape_Area'])
    X = pd.get_dummies(X)  # Aplicar One-Hot Encoding a las características categóricas

    # Imputar valores faltantes
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    y = data['riesgo']

    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Entrenar el modelo SVM
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Evaluar el modelo
    predictions = model.predict(X_test)
    st.write(classification_report(y_test, predictions))

    # Obtener las clases únicas del conjunto de prueba
    unique_classes = y_test.unique()

    # Visualizar la matriz de confusión
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix(y_test, predictions), display_labels=unique_classes).plot(ax=ax, cmap=plt.cm.Blues)
    st.pyplot(fig)

    # Intentar visualizar las características más importantes
    try:
        st.write(f"Shape of model.coef_: {model.coef_.shape}")
        st.write(f"Length of X.columns: {len(X.columns)}")
        st.write(f"Classes: {model.classes_}")

        if model.coef_.shape[1] == len(X.columns):
            feature_importances = pd.DataFrame(model.coef_.T, index=X.columns, columns=model.classes_)
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(feature_importances, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.write("Error: Las dimensiones de las características no coinciden con las del modelo.")
    except Exception as e:
        st.warning(f"{e}")
    

# Función para leer los correos electrónicos de los usuarios registrados desde un archivo CSV
def get_registered_emails(csv_file='data/users.csv'):
    """
    Lee los correos electrónicos de los usuarios registrados desde un archivo CSV.

    Args:
        csv_file (str): El nombre del archivo CSV.

    Returns:
        list: Lista de correos electrónicos de los usuarios registrados.
    """
    try:
        df = pd.read_csv(csv_file)
        emails = df['email'].tolist()
        return emails
    except Exception as e:
        st.error(f"Error al leer los correos electrónicos de los usuarios: {e}")
        return []

# Función para enviar alertas
def send_alert():
    """
    Muestra un formulario para enviar una alerta y envía el correo electrónico de alerta a todos los usuarios registrados cuando se hace clic en el botón 'Enviar'.

    Parameters:
    None

    Returns:
    None
    """
    st.subheader('Enviar Alerta')

    st.info("Por favor, asegúrate de haber autorizado el envío de correos electrónicos desde esta aplicación. También recuerda revisar tu bandeja de spam.")

    alert_type = st.selectbox('Tipo de Alerta', ['Incendio', 'Inundación', 'Terremoto', 'Deslizamiento de tierra'], key='alert_type_selectbox')
    location = st.text_input('Ubicación', key='location_input')
    description = st.text_area('Descripción', key='description_textarea')

    if st.button('Enviar', key='send_alert_button'):
        recipients = get_registered_emails()
        if recipients:
            response = send_email(alert_type, location, description, recipients)
            if response.status_code == 200:
                st.success('Alerta enviada correctamente a todos los usuarios registrados.')
            else:
                st.error(f"Error al enviar la alerta: {response.text}")
        else:
            st.error('No se encontraron correos electrónicos de usuarios registrados.')


# Configuración de la API Key de OpenWeatherMap
openweathermap_api_key = "dde08d392386389bf45f76aee9d97a10"

# Función para obtener datos meteorológicos de OpenWeatherMap
def get_weather(city_name, api_key):
    """
    Retrieves the weather information for a given city using the OpenWeatherMap API.

    Parameters:
    - city_name (str): The name of the city for which to retrieve the weather information.
    - api_key (str): The API key to access the OpenWeatherMap API.

    Returns:
    - dict: A dictionary containing the weather information for the specified city, or None if the information couldn't be retrieved.
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        st.error("No se pudo obtener la información meteorológica.")
        return None

def data_policy():
    st.subheader('Política de Protección de Datos')

    # Texto de la política de datos con caracteres especiales
    data_policy_text = """
    En cumplimiento de la Ley 1581 de 2012 y el Decreto 1377 de 2013, 
    la presente política de protección de datos tiene como finalidad 
    informar a los usuarios sobre el tratamiento de sus datos 
    personales en la plataforma de Gestión de Desastres Naturales en Medellín.

    1. Responsable del Tratamiento de Datos Personales:
    Nombre: Kevin Andrés Blandón Vélez
    Correo electrónico: kblandonv@unal.edu.co

    2. Finalidad del Tratamiento de Datos:
    Los datos personales recolectados serán utilizados para los siguientes fines:
    - Registro y autenticación de usuarios.
    - Administración de datos ingresados relacionados con riesgos naturales.
    - Contacto con los usuarios para fines administrativos y de seguridad.

    3. Derechos de los Titulares:
    Los titulares de los datos personales tienen los siguientes derechos:
    - Conocer, actualizar y rectificar sus datos personales.
    - Solicitar prueba de la autorización otorgada para el tratamiento de sus datos.
    - Ser informados sobre el uso que se ha dado a sus datos personales.
    - Presentar quejas ante la Superintendencia de Industria y Comercio por infracciones a la normativa de protección de datos.

    4. Procedimientos para el Ejercicio de Derechos:
    Para el ejercicio de sus derechos, los titulares pueden contactar al responsable del tratamiento a través del correo electrónico mencionado anteriormente.

    5. Seguridad de la Información:
    La plataforma adopta las medidas necesarias para proteger la seguridad de los datos personales de los usuarios, garantizando la confidencialidad, integridad y disponibilidad de la información.

    Al utilizar la plataforma, los usuarios aceptan la presente política de protección de datos.
    """

    # Mostrar la política de datos
    st.write(data_policy_text)

    # Botones de aceptar y no aceptar
    accepted = st.button("Acepto")
    rejected = st.button("No acepto")

    if accepted:
        st.success("Has aceptado la política de protección de datos.")
    elif rejected:
        st.warning("No has aceptado la política de protección de datos.")



# Funciones de análisis y visualización de datos
def visualize_data(data):
    """
    Visualizes the given data by displaying the head of the data and a histogram of risks.

    Parameters:
    data (pandas.DataFrame): The data to be visualized.

    Returns:
    None
    """
    st.subheader('Visualización de Datos')
    st.write(data.head())

    st.subheader('Histograma de Riesgos')
    fig, ax = plt.subplots()
    data['riesgo'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel('Riesgo')
    ax.set_ylabel('Cantidad')
    st.pyplot(fig)

def analyze_data(data):
    """
    Analyzes the given data and displays statistical summaries and visualizations of the risks.

    Parameters:
    - data: pandas DataFrame containing the data to be analyzed

    Returns:
    None
    """
    st.subheader('Análisis de Datos')
    if not data.empty:
        st.write('Resumen estadístico de los riesgos:')
        st.write(data['riesgo'].describe())

        # Mapear las categorías de riesgo a valores numéricos
        risk_mapping = {
            'Alto riesgo no mitigable': 3,
            'Con condiciones de riesgo': 2,
            'Riesgo bajo': 1,
            'Riesgo medio': 2
        }
        data['riesgo_numerico'] = data['riesgo'].map(risk_mapping)

        st.subheader('Distribución de Riesgos')
        st.write('Distribución de los riesgos numéricos:')
        st.write(data['riesgo_numerico'].describe())

        fig, ax = plt.subplots()
        data['riesgo_numerico'].value_counts().sort_index().plot(kind='bar', ax=ax)
        ax.set_xlabel('Riesgo Numérico')
        ax.set_ylabel('Cantidad')
        st.pyplot(fig)
    else:
        st.warning('No hay datos disponibles para realizar el análisis.')

def visualize_soil_classification(data):
    st.subheader('Clasificación del Suelo')
    if not data.empty:
        st.write('Resumen estadístico de la clasificación del suelo:')
        st.write(data['CLASE_SUELO'].describe())

        st.subheader('Distribución de Clases de Suelo')
        st.write('Distribución de las clases de suelo:')
        st.write(data['CLASE_SUELO'].value_counts())

        # Convertir la columna 'Shape' en tuplas de coordenadas
        data['Shape'] = data['Shape'].apply(lambda x: eval(x))

        # Crear la geometría para la visualización en el mapa
        geometry = [Point(xy) for xy in zip(data['Shape'].apply(lambda x: x[0]), data['Shape'].apply(lambda x: x[1]))]
        gdf_soil = gpd.GeoDataFrame(data, geometry=geometry)

        # Crear el mapa
        fig, ax = plt.subplots()
        gdf_soil.plot(ax=ax, color='green')
        ax.set_title('Clasificación del Suelo en Medellín')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        st.pyplot(fig)
    else:
        st.warning('No hay datos disponibles para visualizar la clasificación del suelo.')



# Función para calcular coordenadas X e Y del tile a partir de latitud y longitud
def lat_lon_to_tile_coords(lat, lon, zoom):
    """
    Converts latitude and longitude coordinates to tile coordinates at a given zoom level.

    Args:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.
        zoom (int): Zoom level.

    Returns:
        tuple: x and y tile coordinates.
    """
    x = int((lon + 180.0) / 360.0 * (2 ** zoom))
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * (2 ** zoom))
    return x, y

# Función para obtener la URL del mapa de precipitaciones
def get_precipitation_map_url(api_key, x, y, zoom):
    """
    Returns the URL for the precipitation map tile based on the provided parameters.

    Parameters:
    - api_key (str): The API key for accessing the OpenWeatherMap API.
    - x (int): The x-coordinate of the tile.
    - y (int): The y-coordinate of the tile.
    - zoom (int): The zoom level of the tile.

    Returns:
    - str: The URL for the precipitation map tile.

    Example:
    >>> api_key = "your_api_key"
    >>> x = 10
    >>> y = 20
    >>> zoom = 5
    >>> get_precipitation_map_url(api_key, x, y, zoom)
    'https://tile.openweathermap.org/map/precipitation_new/5/10/20.png?appid=your_api_key'
    """
    layer = "precipitation_new"
    return f"https://tile.openweathermap.org/map/{layer}/{zoom}/{x}/{y}.png?appid={api_key}"

# Función para visualizar mapa de precipitaciones
def show_precipitation_map():
    latitude = 6.2518
    longitude = -75.5636
    zoom_level = 10
    
    # Crear el mapa centrado en las coordenadas proporcionadas
    map_mde = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)
    
    # Añadir la capa de precipitaciones de OpenWeatherMap
    folium.raster_layers.TileLayer(
        tiles=f"https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={openweathermap_api_key}",
        attr='OpenWeatherMap',
        name='Precipitación',
        overlay=True,
        control=True
    ).add_to(map_mde)
    
    # Añadir el control de capas
    folium.LayerControl().add_to(map_mde)
    
    # Renderizar el mapa en Streamlit
    st_folium(map_mde, width=700, height=500)

def plot_map(data):
    """
    Plots a map showing the distribution of natural risks in Medellín.

    Parameters:
    data (pandas.DataFrame): The data containing the coordinates and other information.

    Returns:
    None
    """
    # Convertir las cadenas de coordenadas en tuplas de números
    data['X'] = data['Shape'].apply(lambda point: ast.literal_eval(point)[0] if isinstance(point, str) else np.nan)
    data['Y'] = data['Shape'].apply(lambda point: ast.literal_eval(point)[1] if isinstance(point, str) else np.nan)

    # Eliminar filas con valores faltantes en X o Y
    data = data.dropna(subset=['X', 'Y'])

    # Convertir las coordenadas X e Y en objetos Point
    geometry = [Point(xy) for xy in zip(data['X'], data['Y'])]
    
    # Crear el GeoDataFrame con la geometría y otros datos
    gdf = gpd.GeoDataFrame(data, geometry=geometry)
    
    # Crear el mapa
    fig, ax = plt.subplots()
    gdf.plot(ax=ax, color='red', markersize=5)
    ax.set_title('Distribución de Riesgos Naturales en Medellín')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.grid(True)
    st.pyplot(fig)  # Mostrar el gráfico en Streamlit



# Interfaz de Streamlit
st.title('Plataforma de Gestión de Desastres Naturales en Medellín')

# Definir el menú principal
menu_options = ['Inicio', 'Registro', 'Iniciar Sesión', 'Actualizar Contraseña', 'Añadir Datos', 'Enviar Alerta', 'Análisis de Datos',
                'Visualización de Datos', 'Clima', 'Mapa Precipitaciones', 'Política de Datos', 'Mitigación de Riesgos', 'Recursos de Emergencia',
                'Clasificación del Suelo', "Entrenar y Evaluar SVM", 'Información de Desarrollador']

# Definir una variable de sesión para almacenar el valor de menu
if 'menu' not in st.session_state:
    st.session_state.menu = 'Inicio'

# Resto del código del menú
menu = st.sidebar.selectbox('Menú', options=menu_options)

# Autenticación de usuarios
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
    st.session_state['username'] = ""
    st.session_state['user_role'] = ""

# Registro de usuarios
if menu == 'Registro':
    st.subheader('Registro de Nuevo Usuario')
    new_username = st.text_input('Nombre de Usuario')
    new_password = st.text_input('Contraseña', type='password')
    email = st.text_input('Correo Electrónico')
    role = st.selectbox('Rol', options=['normal', 'admin'])
    if st.button('Registrar'):
        if new_username and new_password and email:
            save_user(new_username, new_password, email, role)
            st.success('Usuario registrado exitosamente!')
        else:
            st.error('Por favor, complete todos los campos.')
    if st.button('Ver Política de Protección de Datos'):
        menu = 'Política de Datos'

# Inicio de sesión
elif menu == 'Iniciar Sesión':
    st.subheader('Inicio de Sesión')
    username = st.text_input('Nombre de Usuario')
    password = st.text_input('Contraseña', type='password')
    if st.button('Iniciar Sesión'):
        user = authenticate_user(username, password)
        if user:
            st.session_state['authenticated'] = True
            st.session_state['username'] = user['username']
            st.session_state['user_role'] = user['role']
            st.success(f'Bienvenido {user["username"]}!')
        else:
            st.error('Nombre de usuario o contraseña incorrectos.')


elif menu == 'Visualización de Datos':
    if 'authenticated' in st.session_state and st.session_state['authenticated']:
        st.subheader('Visualización de Datos')
        data_type = st.selectbox('Tipo de Datos', options=['Movimientos de Masa', 'Inundaciones'])
        if data_type == 'Movimientos de Masa':
            file_path = 'data/movimientos_de_masa.csv'
            if Path(file_path).exists():
                data = pd.read_csv(file_path)
                visualize_data(data)
                plot_map(data)  # Llamada a la función plot_map()
            else:
                st.error('No hay datos de movimientos de masa disponibles.')
        elif data_type == 'Inundaciones':
            file_path = 'data/inundaciones.csv'
            if Path(file_path).exists():
                data = pd.read_csv(file_path)
                # Manejar valores faltantes en la columna 'Shape'
                data['Shape'].fillna('', inplace=True)
                # Evaluar la columna 'Shape' y extraer las coordenadas X e Y
                data['X'] = data['Shape'].apply(lambda point: ast.literal_eval(point)[0] if isinstance(point, str) else np.nan)
                data['Y'] = data['Shape'].apply(lambda point: ast.literal_eval(point)[1] if isinstance(point, str) else np.nan)
                visualize_data(data)
                plot_map(data)  # Llamada a la función plot_map()
            else:
                st.error('No hay datos de inundaciones disponibles.')
    else:
        st.error('Por favor, inicie sesión para acceder a la visualización de datos.')



# Funcionalidades para usuarios autenticados
if st.session_state['authenticated']:
    if menu == 'Enviar Alerta':
        if is_admin(st.session_state['user_role']):
            send_alert()
        else:
            st.error('Acceso denegado. Debe ser administrador para enviar alertas.')

    elif menu == 'Añadir Datos':
        if is_admin(st.session_state['user_role']):
            st.subheader('Añadir Nuevos Datos')
            data_type = st.selectbox('Tipo de Datos', options=['Movimientos de Masa', 'Inundaciones'])
            codigo = st.text_input('Código')
            nombre = st.text_input('Nombre')
            riesgo = st.selectbox('Riesgo', options=['Bajo', 'Medio', 'Alto'])
            descripcion = st.text_input('Descripción')

            if st.button('Añadir'):
                new_data = pd.DataFrame({
                    'codigo': [codigo],
                    'nombre': [nombre],
                    'riesgo': [riesgo],
                    'descripcion': [descripcion]
                })
                file_path = f'data/{data_type.lower()}.csv'
                new_data.to_csv(file_path, mode='a', header=not Path(file_path).exists(), index=False)
                st.success('Datos añadidos exitosamente!')
        else:
            st.error('Acceso denegado. Debe ser administrador para añadir datos.')
    
    elif menu == 'Actualizar Contraseña':
        st.subheader('Actualizar Contraseña')
        new_password = st.text_input('Nueva Contraseña', type='password')
        if st.button('Actualizar'):
            update_password(st.session_state['username'], new_password)
            st.success('Contraseña actualizada exitosamente!')

    elif menu == 'Análisis de Datos':
        if 'authenticated' in st.session_state and st.session_state['authenticated']:
            st.subheader('Análisis de Datos')
            data_type = st.selectbox('Tipo de Datos', options=['Movimientos de Masa', 'Inundaciones'], key='analysis_data_type_selectbox')
            if data_type == 'Movimientos de Masa':
                file_path = 'data/movimientos_de_masa.csv'
                if Path(file_path).exists():
                    data = pd.read_csv(file_path)
                    analyze_data(data)
                else:
                    st.error('No hay datos de movimientos de masa disponibles.')
            elif data_type == 'Inundaciones':
                file_path = 'data/inundaciones.csv'
                if Path(file_path).exists():
                    data = pd.read_csv(file_path)
                    analyze_data(data)
                else:
                    st.error('No hay datos de inundaciones disponibles.')
        else:
            st.error('Por favor, inicie sesión para acceder al análisis de datos.')


# Funcionalidades accesibles a todos
if menu == 'Mitigación de Riesgos':
    st.subheader('Mitigación de Riesgos en Medellín')
    st.write("""
    La ciudad de Medellín, al igual que otras áreas urbanas, enfrenta diversos riesgos naturales que pueden tener un impacto significativo en la vida de sus habitantes. Es fundamental tomar medidas proactivas para mitigar estos riesgos y proteger a la población. A continuación, se presentan algunos consejos y estrategias de mitigación de riesgos:
    """)
    st.write("""
    - **Concientización y educación pública**: Es crucial educar a la población sobre los riesgos naturales que enfrenta la ciudad y cómo prepararse para ellos. Se pueden organizar charlas, talleres y campañas de sensibilización para informar a la comunidad sobre qué hacer en caso de emergencia.
    """)
    st.write("""
    - **Infraestructura resiliente**: La construcción de infraestructura resiliente es fundamental para reducir el impacto de los desastres naturales. Esto incluye la implementación de códigos de construcción robustos que consideren los riesgos sísmicos, de inundación y de deslizamiento de tierra, así como la inversión en sistemas de drenaje y contención de inundaciones.
    """)
    st.write("""
    - **Monitoreo y alerta temprana**: Establecer sistemas de monitoreo y alerta temprana puede ayudar a detectar y responder rápidamente a eventos naturales peligrosos, como terremotos, deslizamientos de tierra e inundaciones. Es importante que los ciudadanos estén familiarizados con estos sistemas y sepan cómo actuar en caso de recibir una alerta.
    """)
    st.write("""
    - **Planificación urbana y ordenamiento territorial**: Una planificación urbana adecuada puede contribuir significativamente a reducir la vulnerabilidad de la ciudad frente a los riesgos naturales. Esto incluye la identificación de áreas de alto riesgo y la implementación de medidas para limitar la ocupación y el desarrollo en estas zonas.
    """)
    st.write("""
    - **Gestión de residuos y conservación ambiental**: La gestión adecuada de residuos y la conservación de áreas naturales son importantes para prevenir deslizamientos de tierra e inundaciones. La deforestación y la urbanización descontrolada pueden aumentar la vulnerabilidad de la ciudad frente a estos riesgos.
    """)
    st.write("""
    Siguiendo estos consejos y estrategias, Medellín puede fortalecer su resiliencia ante los desastres naturales y proteger la seguridad y el bienestar de sus habitantes.
    """)

elif menu == 'Información de Desarrollador':
    show_developer_info()

elif menu == 'Clima':
        st.subheader('Datos Meteorológicos')
        city_name = st.text_input("Ingrese el nombre de la ciudad")
        if st.button("Obtener Clima"):
            weather_data = get_weather(city_name, openweathermap_api_key)
            if weather_data:
                st.write(f"Clima en {city_name}: {weather_data['weather'][0]['description']}")
                st.write(f"Temperatura: {weather_data['main']['temp']} °C")
                st.write(f"Humedad: {weather_data['main']['humidity']} %")
                st.write(f"Velocidad del Viento: {weather_data['wind']['speed']} m/s")

elif menu == 'Entrenar y Evaluar SVM':
    st.subheader('Entrenar y Evaluar Máquina de Soporte Vectorial (SVM)')

# Explicación del SVM y la interpretación de resultados
    st.subheader("Explicación del Modelo SVM")
    st.markdown("""
    **Máquina de Soporte Vectorial (SVM)**

    Un modelo de Máquina de Soporte Vectorial (SVM) es un algoritmo de aprendizaje supervisado que se utiliza tanto para clasificación como para regresión. En problemas de clasificación, 
    el objetivo de SVM es encontrar un hiperplano en un espacio de características de alta dimensión que separe las clases de datos de la mejor manera posible. 
    Los datos se pueden transformar para encontrar un límite de decisión ideal utilizando diferentes núcleos, en este caso, hemos utilizado un núcleo lineal.

    El modelo de Máquina de Soporte Vectorial (SVM) se ha utilizado para clasificar instancias de riesgo de inundaciones en Medellín en cuatro categorías: 
    "Alto riesgo no mitigable", "Con condiciones de riesgo", "Riesgo medio" y "Riesgo bajo". 

    ### Interpretación de los Resultados del Modelo SVM

    La matriz de confusión proporciona una visualización de las predicciones del modelo SVM en comparación con las verdaderas clases de riesgo. Cada celda de la matriz indica la cantidad 
                de predicciones realizadas por el modelo para cada par de clase verdadera y clase predicha.

    #### Alto riesgo no mitigable:
    - Verdadero Positivos (TP): 13
    - Falsos Positivos (FP): 0
    - Falsos Negativos (FN): 0

    Todas las instancias etiquetadas como "Alto riesgo no mitigable" fueron correctamente clasificadas por el modelo. No hubo falsos positivos ni falsos negativos en esta categoría.

    #### Con condiciones de riesgo:
    - Verdadero Positivos (TP): 28
    - Falsos Positivos (FP): 0
    - Falsos Negativos (FN): 0

    Todas las instancias etiquetadas como "Con condiciones de riesgo" también fueron correctamente clasificadas. Nuevamente, no hubo falsos positivos ni falsos negativos.

    #### Riesgo medio:
    - Verdadero Positivos (TP): 4
    - Falsos Positivos (FP): 0
    - Falsos Negativos (FN): 0

    Todas las instancias etiquetadas como "Riesgo medio" fueron correctamente clasificadas. No hubo falsos positivos ni falsos negativos en esta categoría.

    #### Riesgo bajo:
    - Verdadero Positivos (TP): 1
    - Falsos Positivos (FP): 0
    - Falsos Negativos (FN): 0

    La única instancia etiquetada como "Riesgo bajo" también fue correctamente clasificada, sin falsos positivos ni falsos negativos.

    ### Reporte de Clasificación

    - **Precisión:** Indica la proporción de verdaderos positivos sobre el total de instancias clasificadas como positivas. En este caso, la precisión es 1.00 para todas las clases, lo que significa que todas las predicciones positivas son correctas.
    - **Recall:** Indica la proporción de verdaderos positivos sobre el total de instancias verdaderamente positivas. Un recall de 1.00 para todas las clases significa que el modelo identificó correctamente todas las instancias de cada clase.
    - **F1-Score:** Es la media armónica de la precisión y el recall, proporcionando un balance entre ambos. Un F1-score de 1.00 para todas las clases indica un equilibrio perfecto entre precisión y recall.
    - **Support:** Representa el número de ocurrencias de cada clase en el conjunto de prueba. Las clases "Alto riesgo no mitigable", "Con condiciones de riesgo", "Riesgo bajo" y "Riesgo medio" tienen 13, 28, 4 y 1 instancias respectivamente.

    ### Conclusión
    Esto sugiere que el modelo es extremadamente eficaz para predecir el riesgo de inundación en Medellín, lo que podría ser muy útil para la planificación y la gestión de desastres.
    """)
    
    # Llama a la función train_and_evaluate_svm con tus datos existentes
    train_and_evaluate_svm(avenidas_torrenciales_data)


elif menu == 'Recursos de Emergencia':
    st.subheader('Recursos y Servicios de Emergencia en Medellín')
    st.write("""
    En caso de una emergencia, es fundamental conocer los recursos y servicios disponibles para obtener ayuda y asistencia. A continuación, se presentan algunos de los recursos de emergencia disponibles en la ciudad de Medellín:
    """)
    st.write("""
    - **Líneas de Emergencia**: La ciudad cuenta con líneas telefónicas de emergencia disponibles las 24 horas del día, donde los ciudadanos pueden reportar situaciones de emergencia y solicitar asistencia. Algunos de los números de emergencia más importantes son:
        - Policía Nacional: 123
        - Bomberos: 119
        - Cruz Roja: 132
    """)
    st.write("""
    - **Centros de Atención de Urgencias**: Medellín cuenta con varios centros de atención de urgencias y hospitales equipados para brindar atención médica en caso de emergencia. Es importante conocer la ubicación de estos centros y cómo llegar a ellos en caso de necesidad.
    """)
    st.write("""
    - **Refugios de Emergencia**: En situaciones de desastre, se pueden habilitar refugios de emergencia para albergar a las personas afectadas. Estos refugios proporcionan un lugar seguro donde las personas pueden refugiarse temporalmente y recibir ayuda humanitaria.
    """)
    st.write("""
    - **Evacuación y Rutas de Escape**: Es importante familiarizarse con las rutas de evacuación y los puntos de encuentro designados en caso de evacuación. Conocer estas rutas puede ayudar a las personas a evacuar de manera segura en caso de una emergencia.
    """)
    st.write("""
    Conocer y utilizar estos recursos de emergencia puede marcar la diferencia en la capacidad de respuesta y recuperación de la ciudad frente a desastres naturales u otras situaciones de emergencia.
    """)

elif menu == 'Clasificación del Suelo':
    st.subheader('Información sobre la Clasificación del Suelo en Medellín')
    soil_data = pd.read_csv('data/clasificacion_suelo.csv')
    visualize_soil_classification(soil_data)

elif menu == 'Mapa Precipitaciones':
    show_precipitation_map()

elif menu == 'Política de Datos':
    data_policy()

elif menu == 'Inicio':
    st.subheader('Bienvenido a la Plataforma de Gestión de Desastres Naturales')
    st.write("""
    La Plataforma de Gestión de Desastres Naturales en Medellín es una herramienta diseñada para ayudar a 
    las autoridades, instituciones y ciudadanos a gestionar y responder de manera efectiva ante los desastres 
    naturales que puedan afectar a la ciudad. A través de esta plataforma, los usuarios pueden:
    """)
    st.write("""
    - **Registrarse**: Cree una cuenta para acceder a todas las funcionalidades de la plataforma.
    - **Iniciar Sesión**: Acceda a su cuenta para utilizar las diferentes herramientas y servicios disponibles.
    - **Añadir Datos**: Contribuya con información sobre riesgos naturales y eventos relevantes en la ciudad.
    - **Enviar Alertas**: Comunique eventos o situaciones de emergencia a otros usuarios y autoridades.
    - **Realizar Análisis y Visualización de Datos**: Explore datos, identifique patrones y tome decisiones informadas 
    para la gestión de riesgos.
    """)
    st.write("""
    ¡Explora las diferentes opciones en el menú lateral y comienza a utilizar la plataforma para proteger a Medellín 
    de los desastres naturales!
    """)
