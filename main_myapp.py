import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
from collections import Counter
from ctypes import create_string_buffer
from scipy.optimize import linprog  
import numpy as np
import itertools
import matplotlib.pyplot as plt
import textwrap

from kcal_calculate import kcal_calculate
from kcal_calculate import size_category
from kcal_calculate import age_type_category
from kcal_calculate import bar_print
from kcal_calculate import get_other_nutrient_norms
from kcal_calculate import show_nutr_content
from kcal_calculate import protein_need_calc


# все спсики-------------------------------------------------------------------------

metrics_age_types=["в годах","в месецах"]
gender_types=["Самец", "Самка"]
rep_status_types=["Нет", "Щенность (беременность)", "Период лактации"]
berem_time_types=["первые 4 недедели беременности","последние 5 недель беременности"]
lact_time_types=["1 неделя","2 неделя","3 неделя","4 неделя"]
age_category_types=["Щенки","Взрослые","Пожилые"]
size_types=["Мелкие",  "Средние",  "Крупные", "Очень крупные"]
activity_level_cat_1 = ["Пассивный (гуляеет на поводке менее 1ч/день)", "Средний1 (1-3ч/день, низкая активность)",
                          "Средний2 (1-3ч/день, высокая активность)", "Активный (3-6ч/день, рабочие собаки, например, овчарки)",
                          "Высокая активность в экстремальных условиях (гонки на собачьих упряжках со скоростью 168 км/день в условиях сильного холода)",
                          "Взрослые, склонные к ожирению"]
activity_level_cat_2 = ["Пассивный", "Средний", "Активный"]


other_nutrients_1=["Зола, г","Клетчатка, г","Холестерин, мг","Сахар общее, г"]
other_nutrients_2 = ["Холин, мг","Селен, мкг","Йод, мкг","Пантотеновая кислота, мг","Линолевая кислота, г","Фолиевая кислота, мкг","Альфа-линоленовая кислота, г","Арахидоновая кислота, г","ЭПК (50-60%) + ДГК (40-50%), г"]
other_nutrients=other_nutrients_1+other_nutrients_2

major_minerals=["Кальций, мг","Медь, мг","Железо, мг","Магний, мг","Фосфор, мг","Калий, мг",
                "Натрий, мг","Цинк, мг", "Марганец, мг"]

vitamins=[ "Витамин A, мкг","Витамин E, мг","Витамин Д, мкг","Витамин В1 (тиамин), мг","Витамин В2 (Рибофлавин), мг","Витамин В3 (Ниацин), мг","Витамин В6, мг","Витамин В12, мкг"]


# -------------------------------------------------------------------------------------

st.set_page_config(page_title="Рекомендации по питанию собак", layout="centered")
st.header("Рекомендации по питанию собак")
if "show_result_1" not in st.session_state:
    st.session_state.show_result_1 = False
if "show_result_2" not in st.session_state:
    st.session_state.show_result_2 = False



if "select_reproductive_status" not in st.session_state:
    st.session_state.select_reproductive_status = None


if "select_gender" not in st.session_state:
    st.session_state.select_gender = None
if "show_res_berem_time" not in st.session_state:
                   st.session_state.show_res_berem_time = None
if "show_res_lact_time" not in st.session_state:
                   st.session_state.show_res_lact_time = None
if "show_res_num_pup" not in st.session_state:
                   st.session_state.show_res_num_pup = None 

col1, col0 ,col2, col3 = st.columns([3,1, 3, 2])  # col2 будет посередине
with col1:
       weight = st.number_input("Вес собаки (в кг)", min_value=0.0, step=0.1)
with col2:
    age = st.number_input("Возраст собаки", min_value=0, step=1)
with col3:
    age_metric=st.selectbox("Измерение возроста", metrics_age_types)
gender = st.selectbox("Пол собаки", gender_types)

if gender != st.session_state.select_gender:
            st.session_state.select_gender = gender
            st.session_state.show_result_1 = False
            st.session_state.show_result_2 = False
            st.session_state.select_reproductive_status = False
            st.session_state.show_res_berem_time = False
            st.session_state.show_res_num_pup = False
            st.session_state.show_res_lact_time = False
              

if st.session_state.select_gender == gender_types[1]:
    col1, col2 = st.columns([1, 20])  # col2 будет посередине
    with col2:
        reproductive_status = st.selectbox( "Репродуктивный статус", rep_status_types)
    if reproductive_status != st.session_state.select_reproductive_status:
              st.session_state.select_reproductive_status = reproductive_status
              st.session_state.show_result_1 = False
              st.session_state.show_result_2 = False
          
if st.session_state.select_reproductive_status==rep_status_types[1] and st.session_state.select_gender == gender_types[1]:
  col1, col2 = st.columns([3, 20])  # col2 будет посередине
  with col2:            
       berem_time=st.selectbox("Срок беременности", berem_time_types)   
       if berem_time != st.session_state.show_res_berem_time:
                   st.session_state.show_res_berem_time = berem_time
                   st.session_state.show_result_1 = False
                   st.session_state.show_result_2 = False 

elif st.session_state.select_reproductive_status==rep_status_types[2] and st.session_state.select_gender == gender_types[1]:
    col1, col2 = st.columns([3, 20])  # col2 будет посередине
    with col2:  
                lact_time=st.selectbox("Лактационный период", lact_time_types)  
                num_pup=st.number_input("Количесвто щенков", min_value=0, step=1) 
                if lact_time != st.session_state.show_res_lact_time or num_pup!=st.session_state.show_res_num_pup:
                   st.session_state.show_res_lact_time = lact_time
                   st.session_state.show_res_num_pup = num_pup
                   st.session_state.show_result_1 = False
                   st.session_state.show_result_2 = False 
              


@st.cache_data(show_spinner=False)
def load_data():
    food = pd.read_csv("FINAL_COMBINED.csv")
    disease = pd.read_csv("Disease.csv")
    return food, disease

food_df, disease_df = load_data()

df_standart = pd.read_csv("merge_tab.csv")
proteins=df_standart[df_standart["Type"].isin(["Яйца и Молочные продукты", "Мясо"])]["Ingredient"].tolist()
oils=df_standart[df_standart["Type"].isin([ "Масло и жир"])]["Ingredient"].tolist()
carbonates_cer=df_standart[df_standart["Type"].isin(["Крупы"])]["Ingredient"].tolist()
carbonates_veg=df_standart[df_standart["Type"].isin(["Зелень и специи","Овощи и фрукты"])]["Ingredient"].tolist()
other=df_standart[df_standart["Type"].isin(["Вода, соль и сахар"])]["Ingredient"].tolist()
water=["Вода — Обыкновенный"]
dele = df_standart[df_standart["Standart"].isna()]["Ingredient"].tolist()

stop_words=["Beta-Carotene","With Natural Antioxidant", "Minerals","Digest","Dicalcium Phosphate","L-Carnitine","L-Threonine","Composition:","L-Tryptophan","Chicken Flavor","Manganese Sulfate"
"Hydrolyzed Chicken Flavor", "Monosodium Phosphate","Magnesium Oxide","Powdered Cellulose","Taurine","Mixed Tocopherols For Freshness","Natural Flavor","Potassium Alginate","Sodium Tripolyphosphate",
"Dl-Methionine","Calcium Sulfate","Guar Gum","Betaine","Glyceryl Monostearate","Calcium Chloride","Calcium Lactate","Calcium Gluconate","Natural Flavors","Choline Chloride","Calcium Iodate",
"Dextrose","Zinc Oxide","Copper Sulfate","Ferrous Sulfate","Niacin Supplement","Thiamine Mononitrate","Calcium Pantothenate","Riboflavin Supplement","Biotin",'Pyridoxine Hydrochloride',
"Folic Acid","Disodium Phosphate","Potassium Chloride","Chondroitin Sulfate","Copper Proteinate","Potassium Iodide)","Sodium Pyrophosphate","Sodium Hexametaphosphate","Carrageenan",
"Manganous Oxide","Sodium Selenite","Lipoic Acid","Calcium Carbonate","Vitamin A Supplement","Manganese Sulfate","Derivatives Of Vegetable Origin","Cellulose","Potassium Citrate","Glycerin","Vegetable Protein Extracts",
"Manganese Sulfate","Caramel Color","Citric Acid For Freshness","Brewers Dried Yeast","Soybean Mill Run","Glucosamine Hydrochloride","Vitamin A Supplement","Pork Plasma","Pork Gelatin"]



# Инициализируем состояния
if "step" not in st.session_state:
    st.session_state.step = 0  # 0 — начальное, 1 — после генерации, 2 — после расчета


def classify_breed_size(row):
    w = (row["min_weight"] + row["max_weight"]) / 2
    if w <= 10:
        return "Small Breed"
    elif w <= 25:
        return "Medium Breed"
    else:
        return "Large Breed"

@st.cache_data(show_spinner=False)
def preprocess_disease(df):
    df = df.copy()
    df["breed_size_category"] = df.apply(classify_breed_size, axis=1)
    return df

disease_df = preprocess_disease(disease_df)

@st.cache_data(show_spinner=False)
def preprocess_food(df):
    df = df.copy()
    nutrients = [
        "protein", "fat", "carbohydrate (nfe)", "crude fibre", "calcium",
        "phospohorus", "potassium", "sodium", "magnesium", "vitamin e",
        "vitamin c", "omega-3-fatty acids", "omega-6-fatty acids",
    ]
    for col in nutrients:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("%", "")
            .str.replace("IU/kg", "")
            .str.extract(r"([\d\.]+)")
            .astype(float)
            .fillna(0.0)
        )

    df["combined_text"] = (
        df["ingredients"].fillna("")
        .str.cat(df["key benefits"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["product title"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["product description"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["helpful tips"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["need/preference"].fillna(""), sep=" ", na_rep="")
        .str.cat(df["alternate product recommendation"].fillna(""), sep=" ", na_rep="")
    )
    return df

food_df = preprocess_food(food_df)

# -----------------------------------
# 4) TEXT VECTORIZATION & SVD
# -----------------------------------

@st.cache_resource(show_spinner=False)
def build_text_pipeline(corpus, n_components=100):
    vect = TfidfVectorizer(stop_words="english", max_features=5000)
    X_tfidf = vect.fit_transform(corpus)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)

    return vect, svd, X_reduced

vectorizer, svd, X_text_reduced = build_text_pipeline(food_df["combined_text"], n_components=100)

# -----------------------------------
# 5) CATEGORICAL ENCODING
# -----------------------------------

@st.cache_resource(show_spinner=False)
def build_categorical_encoder(df):
    enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    cats = df[["breed size", "lifestage"]].fillna("Unknown")
    enc.fit(cats)
    return enc, enc.transform(cats)

encoder, X_categorical = build_categorical_encoder(food_df)

# -----------------------------------
# 6) COMBINE FEATURES INTO SPARSE MATRIX
# -----------------------------------

@st.cache_resource(show_spinner=False)
def combine_features(text_reduced, _cat_matrix):
    # Turn dense text_reduced into sparse form
    X_sparse_text = csr_matrix(text_reduced)
    return hstack([X_sparse_text, _cat_matrix])

X_combined = combine_features(X_text_reduced, X_categorical)

# -----------------------------------
# 7) TRAIN RIDGE REGRESSORS FOR NUTRIENTS
# -----------------------------------

@st.cache_resource(show_spinner=False)
def train_nutrient_models(food, _X):
    nutrient_models = {}
    scalers = {}

    nutrients = [
        "protein", "fat", "carbohydrate (nfe)", "crude fibre", "calcium",
        "phospohorus", "potassium", "sodium", "magnesium", "vitamin e",
        "vitamin c", "omega-3-fatty acids", "omega-6-fatty acids",
    ]
    to_scale = {
        "sodium",
        "omega-3-fatty acids",
        "omega-6-fatty acids",
        "calcium",
        "phospohorus",
        "potassium",
        "magnesium",
    }

    for nutrient in nutrients:
        y = food[nutrient].fillna(food[nutrient].median()).values.reshape(-1, 1)
        if nutrient in to_scale:
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y).ravel()
        else:
            scaler = None
            y_scaled = y.ravel()

        X_train, _, y_train, _ = train_test_split(_X, y_scaled, test_size=0.2, random_state=42)

        base = Ridge()
        search = GridSearchCV(
            base,
            param_grid={"alpha": [0.1, 1.0]},
            scoring="r2",
            cv=2,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        nutrient_models[nutrient] = search.best_estimator_
        scalers[nutrient] = scaler

    return nutrient_models, scalers

# **This line must run at import-time** so ridge_models is defined before you use it below:
ridge_models, scalers = train_nutrient_models(food_df, X_combined)

# -----------------------------------
# 8) TRAIN RIDGE CLASSIFIERS FOR INGREDIENT PRESENCE
# -----------------------------------

@st.cache_resource(show_spinner=False)
def train_ingredient_models(food, _X):
    all_ings = []
    for txt in food["ingredients"].dropna():
        tokens = [i.strip().lower() for i in txt.split(",")]
        all_ings.extend(tokens)

    counts = Counter(all_ings)
    frequent = [ing for ing, cnt in counts.items() if cnt >= 5]

    targets = {}
    low = food["ingredients"].fillna("").str.lower()
    for ing in frequent:
        targets[ing] = low.apply(lambda s: int(ing in s)).values

    ing_models = {}
    for ing, y in targets.items():
        clf = RidgeClassifier()
        clf.fit(_X, y)
        ing_models[ing] = clf

    return ing_models, frequent

# **This line must run at import-time** so ingredient_models is defined before you use it below:
ingredient_models, frequent_ingredients = train_ingredient_models(food_df, X_combined)

# -----------------------------------
# 9) DISORDER KEYWORDS DICTIONARY
# -----------------------------------

disorder_keywords = {
    "Inherited musculoskeletal disorders": "joint mobility glucosamine arthritis cartilage flexibility",
    "Inherited gastrointestinal disorders": "digest stomach bowel sensitive diarrhea gut ibs",
    "Inherited endocrine disorders": "thyroid metabolism weight diabetes insulin hormone glucose",
    "Inherited eye disorders": "vision eye retina cataract antioxidant sight ocular",
    "Inherited nervous system disorders": "brain seizure cognitive nerve neuro neurological cognition",
    "Inherited cardiovascular disorders": "heart cardiac circulation omega-3 blood pressure vascular",
    "Inherited skin disorders": "skin allergy itch coat omega-6 dermatitis eczema flaky",
    "Inherited immune disorders": "immune defense resistance inflammatory autoimmune",
    "Inherited urinary and reproductive disorders": "urinary bladder kidney renal urine reproductive",
    "Inherited respiratory disorders": "breath respiratory airway lung cough breathing nasal",
    "Inherited blood disorders": "anemia blood iron hemoglobin platelets clotting hemophilia",
}

# -----------------------------------
# 10) STREAMLIT UI LAYOUT
# -----------------------------------

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


st.sidebar.title("🐶 Smart Dog Diet Advisor")
st.sidebar.write("Select breed + disorder → get personalized food suggestions")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=80)


if "select1" not in st.session_state:
    st.session_state.select1 = None
if "select2" not in st.session_state:
    st.session_state.select2 = None

if "prev_ingr_ranges" not in st.session_state:
    st.session_state.prev_ingr_ranges = []
if "prev_nutr_ranges" not in st.session_state:
    st.session_state.prev_nutr_ranges = {}




if "age_sel" not in st.session_state:
    st.session_state.age_sel = None
if "age_metr_sel" not in st.session_state:
    st.session_state.age_metr_sel = None
if "weight_sel" not in st.session_state:
    st.session_state.weight_sel = None
if "activity_level_sel" not in st.session_state:
    st.session_state.activity_level_sel = None
if "kkal_sel" not in st.session_state:
    st.session_state.kkal_sel = None

breed_list = sorted(disease_df["Breed"].unique())
user_breed = st.selectbox("Порода собаки:", breed_list)

min_weight = disease_df.loc[disease_df["Breed"] == user_breed, "min_weight"].values
max_weight = disease_df.loc[disease_df["Breed"] == user_breed, "max_weight"].values
avg_wight=(max_weight[0]+min_weight[0])/2

size_categ = size_category(avg_wight)
age_type_categ = age_type_category(size_categ, age ,age_metric)


if age!=st.session_state.age_sel or age_metric!=st.session_state.age_metric or weight != st.session_state.weight_sel:
    st.session_state.age_sel=age
    st.session_state.age_metric=age_metric
    st.session_state.weight_sel=weight
    st.session_state.show_result_1 = False
    st.session_state.show_result_2 = False

if age_type_categ==age_category_types[1]:
    activity_level_1 = st.selectbox(
        "Уровень активности", activity_level_cat_1)

elif age_type_categ==age_category_types[2]:
    activity_level_2 = st.selectbox(
        "Уровень активности",activity_level_cat_2)

if age_type_categ==age_category_types[1]:
    if activity_level_1!=st.session_state.activity_level_sel:
        st.session_state.activity_level_sel=activity_level_1
        st.session_state.show_result_1 = False
        st.session_state.show_result_2 = False
        
if age_type_categ==age_category_types[2]:
    if  activity_level_2!=st.session_state.activity_level_sel:
        st.session_state.activity_level_sel=activity_level_2
        st.session_state.show_result_1 = False
        st.session_state.show_result_2 = False

if user_breed:
    info = disease_df[disease_df["Breed"] == user_breed]
    if not info.empty:
        breed_size = info["breed_size_category"].values[0]
        disorders = info["Disease"].unique().tolist()
        selected_disorder = st.selectbox("Заболевание:", disorders)
        disorder_type = info[info["Disease"] == selected_disorder]["Disorder"].values[0]

        if user_breed != st.session_state.select1 or selected_disorder!= st.session_state.select2:
            st.session_state.select1 = user_breed
            st.session_state.select2 = selected_disorder
            st.session_state.show_result_1 = False
            st.session_state.show_result_2 = False
            
        # Первая кнопка
        if st.button("Составить рекомендации"):
            st.session_state.show_result_1 = True
        if st.session_state.show_result_1:
            kcal, formula, page =kcal_calculate(st.session_state.select_reproductive_status, st.session_state.show_res_berem_time, st.session_state.show_res_num_pup ,  st.session_state.show_res_lact_time, 
                                age_type_categ, st.session_state.weight_sel, avg_wight,  st.session_state.activity_level_sel, user_breed, age)
            


            
            
            st.markdown(f"Было рассчитано по формуле")
            st.latex(formula)

            url = "https://europeanpetfood.org/wp-content/uploads/2024/09/FEDIAF-Nutritional-Guidelines_2024.pdf#page=" + page
            st.markdown(f"[Подробнее]({url})")
            if kcal<0:
              kcal=0
            metobolic_energy = st.number_input("Киллокаллории в день", min_value=0.0, step=0.1,  value=round(kcal,1) )
            if st.session_state.kkal_sel!=metobolic_energy:
               st.session_state.kkal_sel=metobolic_energy
               st.session_state.show_result_1 = True
               st.session_state.show_result_2 = False
              
            other_nutrient_norms=get_other_nutrient_norms(st.session_state.kkal_sel, age_type_categ, st.session_state.weight_sel, st.session_state.select_reproductive_status)
                                                          
            # 10.1) Build query vector
            keywords = disorder_keywords.get(disorder_type, selected_disorder).lower()
            kw_tfidf = vectorizer.transform([keywords])
            kw_reduced = svd.transform(kw_tfidf)

            # One-hot for (breed_size, "Adult")
            cat_vec = encoder.transform([[breed_size, "Adult"]])
            kw_combined = hstack([csr_matrix(kw_reduced), cat_vec])

            # 10.2) Predict nutrients
            nutrient_preds = {}
            for nut, model in ridge_models.items():
                pred = model.predict(kw_combined)[0]
                sc = scalers.get(nut)
                if sc:
                    pred = sc.inverse_transform([[pred]])[0][0]
                nutrient_preds[nut] = round(pred, 2)

            # 10.3) Rank ingredients
            ing_scores = {
                ing: clf.decision_function(kw_combined)[0]
                for ing, clf in ingredient_models.items()
            }
            top_ings = sorted(ing_scores.items(), key=lambda x: x[1], reverse=True)[:20]

            prot=sorted([i for i in top_ings if i[0].title() in proteins and i[0].title() not in dele], key=lambda x: x[1], reverse=True)[:1]
            prot = [i.title() for i, _ in prot]
            prot=df_standart[df_standart["Ingredient"].isin(prot)]["Standart"].tolist()

            carb_cer=sorted([i for i in top_ings if i[0].title() in carbonates_cer and i[0].title() not in dele], key=lambda x: x[1], reverse=True)[:1]
            carb_cer = [i.title() for i, _ in carb_cer]
            carb_cer=df_standart[df_standart["Ingredient"].isin(carb_cer)]["Standart"].tolist()

            carb_veg=sorted([i for i in top_ings if i[0].title() in carbonates_veg and i[0].title() not in dele], key=lambda x: x[1], reverse=True)[:1]
            carb_veg = [i.title() for i, _ in carb_veg]
            carb_veg=df_standart[df_standart["Ingredient"].isin(carb_veg)]["Standart"].tolist()


            fat=sorted([i for i in top_ings if i[0].title() in oils and i[0].title() not in dele], key=lambda x: x[1], reverse=True)[:1]
            fat = [i.title() for i, _ in fat]
            fat=df_standart[df_standart["Ingredient"].isin(fat)]["Standart"].tolist()

            oth=sorted([i for i in top_ings[:20] if i[0].title() in other and i[0].title() not in dele], key=lambda x: x[1], reverse=True)[:1]
            if len(oth)>0:
              oth = [i.title() for i, _ in oth]
              oth=df_standart[df_standart["Ingredient"].isin(oth)]["Standart"].tolist()
            else:
              oth=[]
            
            ingredients_finish = [i for i in list(set(prot))+list(set(carb_cer+carb_veg+fat))+list(set(oth+water)) if len(i)>0]
                     
            # 10.5) Display
            st.subheader("🌿 Рекомендуемые ингредиенты")
            for ing in ingredients_finish:
                st.write("• " + ing)
            if len(ingredients_finish)>0:
               
                      # --- Загрузка данных ---
                      df_ingr_all = pd.read_csv('food_ingrediets.csv')
                      cols_to_divide = ['Влага', 'Белки', 'Углеводы', 'Жиры']



                      for col in cols_to_divide+other_nutrients+major_minerals+vitamins:
                        if col !='ЭПК (50-60%) + ДГК (40-50%), г':
                          df_ingr_all[col] = df_ingr_all[col].astype(str).str.replace(',', '.', regex=False)
                          df_ingr_all[col] = pd.to_numeric(df_ingr_all[col], errors='coerce')
                        
                      df_ingr_all['ЭПК (50-60%) + ДГК (40-50%), г'] = df_ingr_all['ЭПК, г']*0.5 + df_ingr_all['ДГК, г']*0.5
                      df_ingr_all[cols_to_divide+other_nutrients+major_minerals+vitamins] = df_ingr_all[cols_to_divide+other_nutrients+major_minerals+vitamins] / 100
                      df_ingr_all['ингредиент и описание'] = df_ingr_all['Ингредиенты'] + ' — ' + df_ingr_all['Описание']
                      

                      proteins=df_ingr_all[df_ingr_all["Категория"].isin(["Яйца и Молочные продукты", "Мясо"])]["ингредиент и описание"].tolist()
                      oils=df_ingr_all[df_ingr_all["Категория"].isin([ "Масло и жир"])]["ингредиент и описание"].tolist()
                      carbonates_cer=df_ingr_all[df_ingr_all["Категория"].isin(["Крупы"])]["ингредиент и описание"].tolist()
                      carbonates_veg=df_ingr_all[df_ingr_all["Категория"].isin(["Зелень и специи","Овощи и фрукты"])]["ингредиент и описание"].tolist()
                      other=df_ingr_all[df_ingr_all["Категория"].isin(["Вода, соль и сахар"])]["ингредиент и описание"].tolist()

                      meat_len=len(set(proteins).intersection(set(ingredients_finish)))

                      
###################################################################################################################################################################
                
                      if "selected_ingredients" not in st.session_state or st.session_state.show_result_1==False:
                          # Преобразуем ingredients_finish в set и сохраняем
                          st.session_state.selected_ingredients = set(ingredients_finish)

                      st.title("🍲 Выбор ингредиентов")
                      for category in df_ingr_all['Категория'].dropna().unique():
                          with st.expander(f"{category}"):
                              df_cat = df_ingr_all[df_ingr_all['Категория'] == category]
                              for ingredient in df_cat['Ингредиенты'].dropna().unique():
                                  df_ing = df_cat[df_cat['Ингредиенты'] == ingredient]
                                  unique_descs = df_ing['Описание'].dropna().unique()
                                  
                                  # Описание, отличное от "Обыкновенный"
                                  non_regular_descs = [desc for desc in unique_descs if desc.lower() != "обыкновенный"]
                                  
                                  if len(unique_descs) == 1 and unique_descs[0].lower() != "обыкновенный":
                                      desc = unique_descs[0]
                                      label = f"{ingredient} — {desc}"
                                      key = f"{category}_{ingredient}_{desc}"
                                      text = f"{ingredient} — {desc}" if desc != "Обыкновенный" else f"{ingredient}"
                                      if st.button(text, key=key):
                                          st.session_state.selected_ingredients.add(label)
                                          st.session_state.show_result_2 = False
                                  
                                  elif non_regular_descs:
                                      # Показываем вложенный expander только если есть НЕ "Обыкновенные"
                                      with st.expander(f"{ingredient}"):
                                          for desc in non_regular_descs:
                                              label = f"{ingredient} — {desc}"
                                              key = f"{category}_{ingredient}_{desc}"
                                              if st.button(f"{desc}", key=key):
                                                  st.session_state.selected_ingredients.add(label)
                                                  st.session_state.show_result_2 = False
                                  
                                  # Можно также отобразить "обыкновенные" кнопкой без вложенного expander (по желанию)
                                  regular_descs = [desc for desc in unique_descs if desc.lower() == "обыкновенный"]
                                  for desc in regular_descs:
                                      label = f"{ingredient} — {desc}"
                                      key = f"{category}_{ingredient}_{desc}_reg"
                                      text = f"{ingredient}"  # Без "Обыкновенный" в кнопке
                                      if st.button(text, key=key):
                                          st.session_state.selected_ingredients.add(label)
                                          st.session_state.show_result_2 = False

                      st.markdown("### ✅ Выбранные ингредиенты:")
                      if "to_remove" not in st.session_state:
                          st.session_state.to_remove = None
                      
                      for i in sorted(st.session_state.selected_ingredients):
                          col1, col2 = st.columns([5, 1])
                          col1.write(i.replace(" — Обыкновенный", ""))
                          if col2.button("❌", key=f"remove_{i}"):
                              st.session_state.to_remove = i
                      
                      if st.session_state.to_remove:
                          st.session_state.selected_ingredients.discard(st.session_state.to_remove)
                          st.session_state.to_remove = None
                          st.rerun()
                      # Пример: доступ к выбранным
                      ingredient_names = list(st.session_state.selected_ingredients)
                      food = df_ingr_all.set_index("ингредиент и описание")[cols_to_divide+other_nutrients+major_minerals+vitamins].to_dict(orient='index')


                      # --- Ограничения по количеству каждого ингредиента ---
                      if ingredient_names:
                          st.subheader("Ограничения по количеству ингредиентов (в % от 100 г):")
                          ingr_ranges = []
                          for ingr in ingredient_names:
                              if ingr in proteins:
                                ingr_ranges.append(st.slider(f"{ingr.replace(" — Обыкновенный", "")}", 0, 100, value=(int(40 / meat_len), int(60 / meat_len))))

                              elif ingr in oils:
                                ingr_ranges.append(st.slider(f"{ingr.replace(" — Обыкновенный", "")}", 0, 100, (1,10)))

                              elif ingr in carbonates_cer:
                                ingr_ranges.append(st.slider(f"{ingr.replace(" — Обыкновенный", "")}", 0, 100, (10,35)))

                              elif ingr in carbonates_veg:
                                ingr_ranges.append(st.slider(f"{ingr.replace(" — Обыкновенный", "")}", 0, 100, (10,25)))
                              elif "Вода" in ingr:
                                ingr_ranges.append(st.slider(f"{ingr.replace(" — Обыкновенный", "")}", 0, 100, (0,30)))
                              elif ingr in other:
                                  ingr_ranges.append(st.slider(f"{ingr.replace(" — Обыкновенный", "")}", 0, 100, (1,3)))


                          # --- Ограничения по нутриентам ---
                          st.subheader("Ограничения по нутриентам:")
                          nutr_ranges = {}
                        
                          needeble_proterin = protein_need_calc(st.session_state.kkal_sel, age_type_categ,  st.session_state.weight_sel, st.session_state.select_reproductive_status, age ,age_metric)
                        
                          nutr_ranges['Влага'] = st.slider(f"{'Влага'}", 0, 100, (70, 85))
                          nutr_ranges['Белки'] = st.slider(f"{'Белки'}", 0, 100, (6,21))
                          nutr_ranges['Углеводы'] = st.slider(f"{'Углеводы'}", 0, 100, (5,10))
                          nutr_ranges['Жиры'] = st.slider(f"{'Жиры'}", 0, 100, (1,15))

                          if ingr_ranges != st.session_state.prev_ingr_ranges:
                                st.session_state.show_result_2 = False
                                st.session_state.prev_ingr_ranges = ingr_ranges.copy()
                            
                            # Проверяем, изменились ли ограничения по нутриентам
                          if nutr_ranges != st.session_state.prev_nutr_ranges:
                                st.session_state.show_result_2 = False
                                st.session_state.prev_nutr_ranges = nutr_ranges.copy()
                          
                          # --- Построение задачи LP ---
                          A = [
                              [food[ing][nutr] if val > 0 else -food[ing][nutr]
                              for ing in ingredient_names]
                              for nutr in nutr_ranges
                              for val in (-nutr_ranges[nutr][0]/100, nutr_ranges[nutr][1]/100)
                          ]
                          b = [
                              val / 100 for nutr in nutr_ranges
                              for val in (-nutr_ranges[nutr][0], nutr_ranges[nutr][1])
                          ]

                          A_eq = [[1 for _ in ingredient_names]]
                          b_eq = [1.0]
                          bounds = [(low/100, high/100) for (low, high) in ingr_ranges]

                          # --- Целевая функция ---
                          st.subheader("Что максимизировать?")
                          selected_maximize = st.multiselect(
                              "Выберите нутриенты для максимизации:",
                              cols_to_divide,
                              default=['Влага',"Белки"]
                          )

                        # Инициализация предыдущего значения
                          if "prev_selected_maximize" not in st.session_state:
                            st.session_state.prev_selected_maximize = ['Влага', 'Белки']
                        
                        # Проверка изменений
                          if selected_maximize != st.session_state.prev_selected_maximize:
                            st.session_state.show_result_2 = False
                            st.session_state.prev_selected_maximize = selected_maximize.copy()
                          f = [-sum(food[i][nutr] for nutr in selected_maximize) for i in ingredient_names]


                          if st.button("🔍 Рассчитать оптимальный состав"):
                            st.session_state.show_result_2 = True
                         
                          if st.session_state.show_result_2:
                              res = linprog(f, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

                              if res.success:
                                  st.success("✅ Решение найдено!")
                                  result = {name: round(val * 100, 2) for name, val in zip(ingredient_names, res.x)}
                                  st.markdown("### 📦 Состав (в граммах на 100 г):")
                                  for name, value in result.items():
                                      st.write(f"{name.replace(" — Обыкновенный", "")}: **{value} г**")

                                  st.markdown("### 💪 Питательная ценность на 100 г:")
                                  nutrients = {
                                      nutr: round(sum(res.x[i] * food[name][nutr] for i, name in enumerate(ingredient_names)) * 100, 2)
                                      for nutr in cols_to_divide
                                  }
                                  for k, v in nutrients.items():
                                      st.write(f"**{k}:** {v} г")
                                  en_nutr_100=3.5*nutrients["Белки"]+8.5*nutrients["Жиры"]+3.5*nutrients["Углеводы"]
                                  st.write(f"**Энергетическая ценность:** {en_nutr_100} ккал")

                                  st.write(f"****")

                                  missing = set()

                                  count_nutr_cont_all = {}
                                  for nutr in other_nutrients + major_minerals + vitamins:
                                      total = 0
                                      for i, name in enumerate(ingredient_names):
                                          if nutr not in food[name]:
                                              missing.add((name, nutr))
                                          total += res.x[i] * food[name].get(nutr, 0)
                                      count_nutr_cont_all[nutr] = round(total * 100, 2)
                                  
                                  if missing:
                                      st.warning(f"Отсутствуют значения для: {missing}")



                                  count_nutr_cont_all = {
                                      nutr: round(sum(res.x[i] * food[name][nutr] for i, name in enumerate(ingredient_names)) * 100, 2)
                                      for nutr in other_nutrients+major_minerals+vitamins
                                  }


                                  st.markdown(f"### Сколько нужно в граммах корма и ингредиентов на {round(metobolic_energy,1)} ккал")           
                                  needed_feed_g = (metobolic_energy * 100) / en_nutr_100
                                  ingredients_required = {
                                      name: round((weight * needed_feed_g / 100), 2)
                                      for name, weight in result.items()
                                  }                                  
                                  st.write(f"📌 Корм: {round(needed_feed_g, 2)} г")
                                  st.write("🧾 Количество ингредиентов для этой порции:")
                                  for ingredient, amount in ingredients_required.items():
                                      st.write(f" - {ingredient.replace(" — Обыкновенный", "")}: {amount} г")

                                
                                  count_nutr_cont_all = {
                                      nutr: round(sum(amount * food[ingredient][nutr] for ingredient, amount in ingredients_required.items()), 2)
                                      for nutr in cols_to_divide+other_nutrients+major_minerals+vitamins
                                  }

                                  st.markdown(f"### 💪 Питательная ценность на {round(needed_feed_g, 2)} г:")

                                  for k in cols_to_divide:
                                      st.write(f"**{k}:** {count_nutr_cont_all[k]} г")
                                  st.write(f"****") 
                                
                                  show_nutr_content(count_nutr_cont_all, other_nutrient_norms)    
                                
                            
                              else:
                                  st.error("❌ Не удалось найти оптимальное решение. Попробуйте другие параметры.")
                                  with st.spinner("🔄 Ищем по другому методу..."):
                            
                                        step = 1  # шаг в процентах
                                        variants = []
                                        ranges = [np.arange(low, high + step, step) for (low, high) in ingr_ranges]
                            
                                        # Генерация всех комбинаций, которые дают в сумме 100 г
                                        for combo in itertools.product(*ranges):
                                            if abs(sum(combo) - 100) < 1e-6:
                                                variants.append(combo)
                            
                                        best_recipe = None
                                        min_penalty = float("inf")
                            
                                        for combo in variants:
                                            values = dict(zip(ingredient_names, combo))
                            
                                            totals = {nutr: 0.0 for nutr in cols_to_divide}
                                            for i, ingr in enumerate(ingredient_names):
                                                for nutr in cols_to_divide:
                                                    totals[nutr] += values[ingr] * food[ingr][nutr]
                            
                                            # Штраф за отклонения от допустимых диапазонов
                                            penalty = 0
                                            for nutr in cols_to_divide:
                                                val = totals[nutr]
                                                min_val = nutr_ranges[nutr][0]
                                                max_val = nutr_ranges[nutr][1]
                            
                                                if val < min_val:
                                                    penalty += min_val - val
                                                elif val > max_val:
                                                    penalty += val - max_val
                            
                                            if penalty < min_penalty:
                                                min_penalty = penalty
                                                best_recipe = (values, totals)
                    
                                  if best_recipe:
                                    values, totals = best_recipe
                                    st.success("⚙️ Найден состав перебором:")
                    
                                    st.markdown("### 📦 Состав (в граммах на 100 г):")
                                    for name, val in values.items():
                                        st.write(f"{name.replace(" — Обыкновенный", "")}: **{round(val, 2)} г**")
 
                                    
                                    st.markdown("### 💪 Питательная ценность на 100 г:")
                                    for nutr in cols_to_divide:
                                        st.write(f"**{nutr}:** {round(totals[nutr], 2)} г")
                                   
                                    en_nutr_100=3.5*totals["Белки"]+8.5*totals["Жиры"]+3.5*totals["Углеводы"]
                                    st.write(f"**Энергетическая ценность:** {round(en_nutr_100,2)} ккал")



                                    
                                    st.markdown(f"### Сколько нужно в граммах корма и ингредиентов на {round(metobolic_energy,1)} ккал")           
                                    needed_feed_g = (metobolic_energy * 100) / en_nutr_100
                                    ingredients_required = {
                                        name: round((weight * needed_feed_g / 100), 2)
                                        for name, weight in values.items()
                                    }                                  
                                    st.write(f"📌 Корм: {round(needed_feed_g, 2)} г")
                                    st.write("🧾 Количество ингредиентов для этой порции:")
                                    for ingredient, amount in ingredients_required.items():
                                        st.write(f" - {ingredient.replace(" — Обыкновенный", "")}: {amount} г")

                                    count_nutr_cont_all = {
                                      nutr: round(sum(amount * food[ingredient][nutr] for ingredient, amount in ingredients_required.items()), 2)
                                      for nutr in cols_to_divide+other_nutrients+major_minerals+vitamins }
                                    

                                    st.markdown(f"### 💪 Питательная ценность на {round(needed_feed_g, 2)} г:")

                                    for k in cols_to_divide:
                                      st.write(f"**{k}:** {count_nutr_cont_all[k]} г")
                                    st.write(f"****") 
                                    show_nutr_content(count_nutr_cont_all, other_nutrient_norms)   






                                    
                                    # --- График 1: Состав ингредиентов ---
                                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                                    
                                    ingr_vals = [values[i] for i in ingredient_names]
                                    ingr_lims = ingr_ranges
                                    
                                    lower_errors = [val - low for val, (low, high) in zip(ingr_vals, ingr_lims)]
                                    upper_errors = [high - val for val, (low, high) in zip(ingr_vals, ingr_lims)]
                                    
                                    wrapped_ingredients = ['\n'.join(textwrap.wrap(label.replace(" — Обыкновенный", ""), 10)) for label in ingredient_names]
                                    
                                    ax1.errorbar(wrapped_ingredients, ingr_vals, yerr=[lower_errors, upper_errors],
                                                 fmt='o', capsize=5, color='#FF4B4B', ecolor='#1E90FF', elinewidth=2)
                                    ax1.set_ylabel("Значение")
                                    ax1.set_title("Ингредиенты: значения и ограничения")
                                    ax1.set_ylim(0, 100)
                                    ax1.grid(True, axis='y', linestyle='-', color='#e6e6e6', alpha=0.7)
                                    ax1.tick_params(axis='x', rotation=0)
                                    ax1.spines['top'].set_color('white')
                                    ax1.spines['right'].set_visible(False)
                                    
                                    st.pyplot(fig1)
                                    
                                    # --- График 2: Питательные вещества ---
                                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                                    
                                    nutrients = list(nutr_ranges.keys())
                                    nutr_vals = [totals[n] for n in nutrients]
                                    nutr_lims = [nutr_ranges[n] for n in nutrients]
                                    
                                    for i, (nutrient, val, (low, high)) in enumerate(zip(nutrients, nutr_vals, nutr_lims)):
                                        ax2.plot([i, i], [low, high], color='#1E90FF', linewidth=4, alpha=0.5)
                                        ax2.plot(i, val, 'o', color='#FF4B4B')
                                    
                                    ax2.set_xticks(range(len(nutrients)))
                                    ax2.set_xticklabels(nutrients, rotation=0)
                                    ax2.set_ylabel("Значение")
                                    ax2.set_title("Питательные вещества: значения и допустимые границы")
                                    ax2.set_ylim(0, 100)
                                    ax2.grid(True, axis='y', linestyle='-', color='#e6e6e6', alpha=0.7)
                                    ax2.spines['top'].set_color('white')
                                    ax2.spines['right'].set_visible(False)
                                    
                                    st.pyplot(fig2)
                                 
                                  else:
                                     st.error("🚫 Не удалось найти подходящий состав даже вручную.")

            
           

                      else:
                          st.info("👈 Пожалуйста, выберите хотя бы один ингредиент.")
