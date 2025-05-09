import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from catboost import CatBoostClassifier
import tensorflow as tf
import os

st.set_page_config(page_title="ML", layout="centered", initial_sidebar_state="expanded")

with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

file_path = r'C:\Users\Жанна\Desktop\OMGTU\ML_RGR\ML_RGR\data\final_data_card_transdata.csv'
data = pd.read_csv(file_path)

output_dir = r'C:\Users\Жанна\Desktop\OMGTU\ML_RGR\ML_RGR\rgr_models'
scaler_path = os.path.join(output_dir, 'scaler.pkl')

try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Ошибка загрузки scaler: {str(e)}")
    scaler = None

def load_model(model_name, method='pickle'):
    model_path = os.path.join(output_dir, f'{model_name}.{method}' if method != 'tensorflow' else f'{model_name}.keras')
    try:
        if method == 'pickle':
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        elif method == 'catboost':
            model = CatBoostClassifier()
            model.load_model(model_path)
            return model
        elif method == 'tensorflow':
            return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Ошибка загрузки модели {model_name}: {str(e)}")
        return None
    
models = {}
models['kNN (Manhattan)'] = load_model('knn_manhattan', method='pickle')
models['Gradient Boosting'] = load_model('gradient_boosting', method='pickle')
models['CatBoost'] = load_model('catboost', method='catboost')
models['Bagging'] = load_model('bagging', method='pickle')
models['Stacking'] = load_model('stacking', method='pickle')
models['Neural Network'] = load_model('neural_network', method='tensorflow')

available_models = {k: v for k, v in models.items() if v is not None}
if not available_models:
    st.error("Не удалось загрузить ни одну модель. Проверьте файлы моделей.")
    st.stop()


with st.sidebar:
    st.markdown("<h2 style='color:#3f3f3f;'>Меню</h2>", unsafe_allow_html=True)
    page = st.radio(
        "Выберите страницу",
        ["О разработчике", "О наборе данных", "Визуализация", "Инференс моделей"],
        label_visibility="collapsed"
    )


if page == "О разработчике":
    st.markdown("<div class='title'>🐱‍💻 Информация о разработчике</div>", unsafe_allow_html=True)

    st.markdown("""<hr style="border-color:#333;">""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2], gap="medium")

    with col1:
        st.image("photo.jpg", width=280)

    with col2:
        st.markdown("""
            <div class='info-card'>
                <div class='subtitle'>🐾Личная информация</div>
                <div class='info'> <u>ФИО</u>: Смоловая Жанна Евгеньевна</div>
                <div class='info'><u>Номер учебной группы:</u> МО-231</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='subtitle'>📚 Тема работы</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='info'>
            Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных
        </div>
    """, unsafe_allow_html=True)

elif page == "О наборе данных":
    st.markdown("<div class='title'>📼 О наборе данных</div>", unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <div class='subtitle'>💡 Предметная область</div>
            <div class='info'>
                Датасет представляет собой данные о транзакциях с кредитными картами, целью которых является обнаружение мошеннических операций. 
                Каждая запись описывает одну транзакцию, а целевая переменная указывает, является ли транзакция мошеннической (1) или нет (0). 
                Этот набор данных полезен для разработки моделей машинного обучения, направленных на повышение безопасности финансовых операций.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <div class='subtitle'>🔍 Описание признаков</div>
            <div class='info'>
                <i><strong>distance_from_home</strong></i>: Расстояние от дома клиента (км).<br>
                <i><strong>distance_from_last_transaction</strong></i>: Расстояние от последней транзакции (км).<br>
                <i><strong>repeat_retailer</strong></i>: Индикатор повторного ритейлера.<br>
                <i><strong>used_chip</strong></i>: Использование чипа при транзакции.<br>
                <i><strong>used_pin_number</strong></i>: Использование PIN-кода при транзакции.<br>
                <i><strong>online_order</strong></i>: Транзакция совершена онлайн.<br>
                <i><strong>transaction_speed</strong></i>: Время, затраченное на выполнение транзакции (сек).<br>
                <i><strong>secure_online_transaction</strong></i>: Индикатор защищённой онлайн-транзакции.<br>
                <i><strong>fraud</strong></i>: Целевая переменная, мошенническая транзакция.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <div class='subtitle'>🛠️ Особенности предобработки</div>
            <div class='info'>
                - Удалены признаки с высокой корреляцией: <i><strong>ratio_to_median_purchase_price</strong></i> (корреляция 0.91) и <i><strong>high_price_flag</strong></i> (корреляция 0.62), чтобы избежать утечки данных.<br>
                - Данные сбалансированы с помощью метода <i>SMOTE</i>, так как изначально классы были несбалансированы (большинство транзакций — не мошеннические).<br>
                - Применена нормализация с использованием <i>StandardScaler</i> для приведения признаков к единому масштабу.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section'>
            <div class='subtitle'>📈 Особенности EDA</div>
            <div class='info'>
                - Анализ распределения классов показал сильную несбалансированность: около 90% транзакций — не мошеннические.<br>
                - Корреляционный анализ выявил высокую корреляцию между признаками <i><strong>ratio_to_median_purchase_price</strong></i>, <i><strong>high_price_flag</strong></i> и целевой переменной, что привело к их удалению.<br>
                - Выбросы в признаке <i><strong>distance_from_home</strong> были исследованы: некоторые транзакции совершались на расстоянии более 1000 км от дома, что может быть индикатором мошенничества.<br>
                - Распределение признаков, таких как <i><strong>used_pin_number</strong></i> и <i><strong>online_order</strong></i>, показало, что мошеннические транзакции чаще происходят онлайн и без использования PIN-кода.
            </div>
        </div>
    """, unsafe_allow_html=True)

elif page == "Визуализация":
    st.markdown("<div class='title'>📉 Визуализация</div>", unsafe_allow_html=True)

    st.markdown("<div class='subtitle'>📊 Гистограмма: Распределение расстояния от дома</div>", unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x="distance_from_home", bins=50, color="#ff7e05")
    ax1.set_title("Распределение расстояния от дома")
    ax1.set_xlabel("Расстояние (км)")
    ax1.set_ylabel("Частота")
    plt.tight_layout()
    st.pyplot(fig1)

    st.markdown("<div class='subtitle'>📏 Boxplot: Расстояние от последней транзакции по классам</div>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x="fraud", y="distance_from_last_transaction", palette={"0": "#fac761", "1": "#ff7e05"})
    ax2.set_title("Расстояние от последней транзакции по классам")
    ax2.set_xlabel("Мошенничество (0: Нет, 1: Да)")
    ax2.set_ylabel("Расстояние (км)")
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("<div class='subtitle'>📊 Countplot: Распределение онлайн-заказов по классам</div>", unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=data, x="online_order", hue="fraud", palette="Oranges")
    ax3.set_title("Распределение онлайн-заказов по классам")
    ax3.set_xlabel("Онлайн-заказ (0: Нет, 1: Да)")
    ax3.set_ylabel("Количество")
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("<div class='subtitle'>🌡️ Heatmap: Корреляционная матрица</div>", unsafe_allow_html=True)
    numeric_data = data.select_dtypes(include=[np.number])
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="YlOrRd", vmin=-1, vmax=1)
    ax4.set_title("Корреляционная матрица")
    plt.tight_layout()
    st.pyplot(fig4)

elif page == "Инференс моделей":
    st.markdown("<div class='title'>💻 Инференс моделей</div>", unsafe_allow_html=True)

    st.markdown("<div class='subtitle'>Предсказание мошенничества</div>", unsafe_allow_html=True)

    model_choice = st.selectbox("Выберите модель для предсказания", list(available_models.keys()))

    uploaded_file = st.file_uploader("📂 Загрузите CSV файл с данными", type="csv")
    
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        required_columns = ['distance_from_home', 'distance_from_last_transaction', 'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order', 'transaction_speed', 'secure_online_transaction']
        if all(col in input_data.columns for col in required_columns):
            input_data = input_data[required_columns]
        else:
            st.error("CSV файл должен содержать столбцы: " + ", ".join(required_columns))
            input_data = None
    else:
        st.markdown("<div class='info'>Или введите данные вручную:</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            distance_from_home = st.number_input("Расстояние от дома (км)", min_value=0.0, value=5.0, step=0.1)
            distance_from_last_transaction = st.number_input("Расстояние от последней транзакции (км)", min_value=0.0, value=2.0, step=0.1)
            repeat_retailer = st.selectbox("Повторный ритейлер", [0, 1])
            used_chip = st.selectbox("Использование чипа", [0, 1])
        with col2:
            used_pin_number = st.selectbox("Использование PIN", [0, 1])
            online_order = st.selectbox("Онлайн-заказ", [0, 1])
            transaction_speed = st.number_input("Скорость транзакции (сек)", min_value=0.0, value=0.1, step=0.01)
            secure_online_transaction = st.selectbox("Безопасная онлайн-транзакция", [0, 1])

        if distance_from_home < 0 or distance_from_last_transaction < 0 or transaction_speed < 0:
            st.error("Расстояние и скорость не могут быть отрицательными!")
        else:
            input_data = pd.DataFrame({
                'distance_from_home': [distance_from_home],
                'distance_from_last_transaction': [distance_from_last_transaction],
                'repeat_retailer': [repeat_retailer],
                'used_chip': [used_chip],
                'used_pin_number': [used_pin_number],
                'online_order': [online_order],
                'transaction_speed': [transaction_speed],
                'secure_online_transaction': [secure_online_transaction]
            })

    if st.button("🚀 Получить предсказание") and input_data is not None and scaler is not None:
        try:
            input_scaled = scaler.transform(input_data)
            model = available_models[model_choice]

            if model_choice == 'Neural Network':
                prediction = (model.predict(input_scaled, verbose=0) > 0.5).astype(int)[0]
                probability = float(model.predict(input_scaled, verbose=0)[0])  # Преобразуем в float
            else:
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]

            result_class = 'result-safe' if prediction == 0 else 'result-fraud'
            st.markdown(f"<div class='{result_class}'><u>Результат:</u> {'Мошенничество' if prediction == 1 else 'Не мошенничество'}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='{result_class}'><u>Вероятность мошенничества:</u> {probability:.2%}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Ошибка при предсказании: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)