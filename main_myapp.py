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
from kcal_calculate import classify_breed_size
import sqlite3

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

main_nutrs=['moisture_per', 'protein_per', 'carbohydrate_per', 'fats_per']
other_nutrients_1=['ash_g', 'fiber_g', 'cholesterol_mg', 'total_sugar_g']
other_nutrients_2 = ['choline_mg', 'selenium_mcg', 'iodine_mcg', 'linoleic_acid_g','alpha_linolenic_acid_g', 'arachidonic_acid_g', 'epa_g', 'dha_g']
other_nutrients=other_nutrients_1+other_nutrients_2

major_minerals=['calcium_mg', 'phosphorus_mg', 'magnesium_mg', 'sodium_mg', 'potassium_mg', 'iron_mg', 'copper_mg', 'zinc_mg', 'manganese_mg']
vitamins=['vitamin_a_mcg', 'vitamin_e_mg', 'vitamin_d_mcg', 'vitamin_b1_mg', 'vitamin_b2_mg', 'vitamin_b3_mg', 'vitamin_b5_mg', 'vitamin_b6_mg', 'vitamin_b9_mcg', 'vitamin_b12_mcg', 'vitamin_c_mg', 'vitamin_k_mcg']

disorder_keywords = {
    "Inherited musculoskeletal disorders": "muscle joint bone cartilage jd joint mobility glucosamine arthritis cartilage flexibility",
    "Inherited gastrointestinal disorders": "digestive digestion stool food sensitivity hypoallergenic stomach digest stomach bowel sensitive diarrhea gut ibs",
    "Inherited endocrine disorders": "thyroid metabolism weight diabetes insulin hormone glucose",
    "Inherited eye disorders": "vision eye retina cataract antioxidant sight ocular",
    "Inherited nervous system disorders": "nervous system stress disrupted sleep brain brain seizure cognitive nerve neuro neurological cognition",
    "Inherited cardiovascular disorders": "heart hd heart cardiac circulation omega-3 blood pressure vascular",
    "Inherited skin disorders": "skin coat allergy skin allergy itch coat omega-6 dermatitis eczema flaky",
    "Inherited immune disorders": "immune defense resistance inflammatory autoimmune",
    "Inherited urinary and reproductive disorders": " urinary bladder stones urinary bladder kidney renal urine reproductive",
    "Inherited respiratory disorders": "breath respiratory airway lung cough breathing nasal",
    "Inherited blood disorders": "anemia blood iron hemoglobin platelets clotting hemophilia",
	"aging care":"aging senior mature",
	"puppy care":"puppy grow start",
	"adult care":"adult immune optimal delicious",
	"weight management":"weight management overweight",
	"food sensitivity":"food sensitivity hypoallergenic stomach"	
}


transl_dis={
 "Inherited musculoskeletal disorders": ["musculoskeletal and joint care"] ,
    "Inherited gastrointestinal disorders": ["digestive care","food sensitivity"],
    "Inherited endocrine disorders": ["weight management"],
    "Inherited eye disorders": ["nervous system care and stress"],
    "Inherited nervous system disorders": ["nervous system care and stress"],
    "Inherited cardiovascular disorders": ["heart care"],
    "Inherited skin disorders": ["skin health"],
    "Inherited immune disorders": ["aging care","puppy care","adult care"]	,
    "Inherited urinary and reproductive disorders": ["urinary care"],
    "Inherited respiratory disorders": ["aging care","puppy care","adult care"]	,
    "Inherited blood disorders" : ["aging care","puppy care","adult care"],
	"aging care":["aging care"],
	"puppy care":["puppy care"],
	"adult care":["adult care"],
	"weight management":["weight management"],
	"food sensitivity":["food sensitivity"]
}

transl_size={"Мелкие":"small",  "Средние":"medium", 	"Крупные":"large", "Очень крупные":"large"}

transl_age={"Щенки":"puppy","Взрослые":"adult","Пожилые":"senior"}

transl_nutrs={
	"moisture":'Влага', 
    "protein":'Белки', 
    "fat":'Жиры', 
    "carbohydrate":'Углеводы'}
# загрузка и подготовка датасетов-------------------------------------------------------------------------------------

from scipy.sparse import csr_matrix
import numpy as np

def apply_category_masks(X, encoder):
    X = X.toarray()
    feature_names = encoder.get_feature_names_out()

    idx = {name: i for i, name in enumerate(feature_names)}

    # пример: breed_size = "-"
    if "breed_size_-" in idx:
        mask = X[:, idx["breed_size_-"]] == 1
        for k in ["breed_size_s", "breed_size_m", "breed_size_l"]:
            if k in idx:
                X[mask, idx[k]] = 1

    # life_stage = "-"
    if "life_stage_-" in idx:
        mask = X[:, idx["life_stage_-"]] == 1
        for k in ["life_stage_puppy", "life_stage_adult", "life_stage_senior"]:
            if k in idx:
                X[mask, idx[k]] = 1

    return csr_matrix(X)

@st.cache_data(show_spinner=False)
def load_data():
    conn = sqlite3.connect("pet_food.db")
    food=pd.read_sql("""SELECT name_product, description, ingredients, GROUP_CONCAT(category.category) AS category,
food_form.food_form,  breed_size.breed_size,  life_stage.life_stage, 
moisture, protein, fat as fats, carbohydrate FROM dog_food 
INNER JOIN dog_food_characteristics ON dog_food_characteristics.id_dog_food = dog_food.id_dog_food 
INNER JOIN breed_size ON dog_food_characteristics.id_breed_size = breed_size.id_breed_size
INNER JOIN life_stage ON dog_food_characteristics.id_life_stage = life_stage.id_life_stage 
INNER JOIN food_form ON dog_food_characteristics.id_food_form = food_form.id_food_form 
INNER JOIN food_category_connect ON food_category_connect.id_dog_food = dog_food.id_dog_food 
INNER JOIN category ON food_category_connect.id_category = category.id_category 
INNER JOIN nutrient_macro ON nutrient_macro.id_dog_food = dog_food.id_dog_food 
GROUP BY dog_food.id_dog_food""", conn)
	
    food["category"] = (food["category"].astype(str).str.split(", "))

    conn= sqlite3.connect("dog_breed_disease.db")
    disease = pd.read_sql("""SELECT breed_name.name_ru as name_breed,  min_weight, max_weight, disease.name_ru as name_disease, name_disorder
                FROM breed 
                inner join breed_name on breed.id_breed = breed_name.id_breed
                inner join breed_disease on breed.id_breed = breed_disease.id_breed
                inner join disease on disease.id_disease= breed_disease.id_disease
                inner join disease_disorder on disease.id_disease= disease_disorder.id_disease
                inner join disorder on disorder.id_disorder=disease_disorder.id_disorder""", conn)
	
    disease["breed_size_category"] = disease.apply(classify_breed_size, axis=1)
    conn=sqlite3.connect("ingredients.db")
    standart = pd.read_sql("""SELECT name_feed_ingredient,  ingredients_translation.name_ru || " — " || format_ingredients_translation.name_ru AS ingredient_full_ru, ingredient_category.name_ru as category_ru     
FROM  ingredient_mapping
inner join ingredient on ingredient.id_ingredient	= ingredient_mapping.id_ingredient
inner join ingredients_translation on ingredients_translation.id_ingredient_name=ingredient.id_ingredient_name
inner join format_ingredients_translation on format_ingredients_translation.id_format_ingredient = ingredient.id_format_ingredient
inner join ingredient_category on ingredient_category.id_category = ingredient.id_category""", conn)

    ingredirents_df =  pd.read_sql("""SELECT format_ingredient, ingredients_translation.name_ru as name_ingredient_ru , format_ingredients_translation.name_ru as format_ingredient_ru, ingredient_category.name_ru as category_ru, 

                      ingredients_translation.name_ru || " — " || format_ingredients_translation.name_ru AS ingredient_format_cat,

                      calories_kcal, moisture_per, protein_per, carbohydrate_per,fats_per, ash_g, fiber_g, cholesterol_mg, total_sugar_g,
                      
                      calcium_mg, phosphorus_mg, magnesium_mg, sodium_mg, potassium_mg, iron_mg, copper_mg, zinc_mg, manganese_mg, selenium_mcg, iodine_mcg, choline_mg,
                      
                      vitamin_a_mcg,  vitamin_e_mg,  vitamin_d_mcg, vitamin_b1_mg, vitamin_b2_mg,vitamin_b3_mg, 
                      vitamin_b5_mg, vitamin_b6_mg,vitamin_b9_mcg,vitamin_b12_mcg, vitamin_c_mg, vitamin_k_mcg,
                      alpha_carotene_mcg,beta_carotene_mcg, beta_cryptoxanthin_mcg, lutein_zeaxanthin_mcg, lycopene_mcg, retinol_mcg, 
                      linoleic_acid_g, alpha_linolenic_acid_g , arachidonic_acid_g ,epa_g, dha_g
                      
                      FROM  ingredient
                      inner join ingredients_translation on ingredient.id_ingredient_name=ingredients_translation.id_ingredient_name
                      inner join format_ingredients_translation on format_ingredients_translation.id_format_ingredient=ingredient.id_format_ingredient
                      inner join ingredient_category on ingredient_category.id_category= ingredient.id_category

                      inner join nutrient_macro on nutrient_macro.id_ingredient=ingredient.id_ingredient
                      inner join nutrient_micro on nutrient_micro.id_ingredient=ingredient.id_ingredient
                      inner join vitamin on vitamin.id_ingredient=ingredient.id_ingredient
                      inner join vitamin_a_related_compounds on vitamin_a_related_compounds.id_ingredient=ingredient.id_ingredient
                      inner join fatty_acids on fatty_acids.id_ingredient=ingredient.id_ingredient""", conn)
    nutrients_transl= pd.read_sql("""SELECT name_in_database, name_ru FROM  nutrients_names """, conn)

    return food, disease, standart, ingredirents_df,nutrients_transl

food_df, disease_df, df_standart, ingredirents_df,nutrients_transl= load_data()

proteins=df_standart[df_standart["category_ru"].isin(["Мясо","Яйца и Молочные продукты"])]["name_feed_ingredient"].tolist()
oils=df_standart[df_standart["category_ru"].isin([ "Масло и жир"])]["name_feed_ingredient"].tolist()
carbonates_cer=df_standart[df_standart["category_ru"].isin(["Крупы"])]["name_feed_ingredient"].tolist()
carbonates_veg=df_standart[df_standart["category_ru"].isin(["Зелень и специи","Овощи и фрукты"])]["name_feed_ingredient"].tolist()
water=["water"]


#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Расчеты и обучение модели для рекомендации ингредиентов-------------------------------------------------------------------------------------------------
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

vectorizer, svd, X_text_reduced = build_text_pipeline(food_df["description"], n_components=100)

# -----------------------------------
# 5) CATEGORICAL ENCODING
# -----------------------------------

@st.cache_resource(show_spinner=False)
def build_categorical_encoder(df):
    cats = df[["breed_size", "life_stage"]]

    enc = OneHotEncoder(
        sparse_output=True,
        handle_unknown="ignore"
    )

    enc.fit(cats)
    X = enc.transform(cats)

    return enc, X

encoder, X_categorical = build_categorical_encoder(food_df)

X_categorical=apply_category_masks(X_categorical,encoder)

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
# 7) TRAIN RIDGE CLASSIFIERS FOR INGREDIENT PRESENCE
# -----------------------------------

@st.cache_resource(show_spinner=False)
def train_ingredient_models(food, _X):
    parsed_ings = []
    for txt in food["ingredients"].dropna():
        tokens = (txt.split(", ") )
        parsed_ings.append(set(tokens))

    # --- 2) Список уникальных ингредиентов ---
    all_ings = [ing for s in parsed_ings for ing in s]
    frequent = list(set(all_ings))

    # --- 3) Формирование бинарных таргетов ---
    targets = {}
    parsed_series = food["ingredients"].fillna("").apply(
        lambda txt: set(txt.split(", ")) if txt else set())

    for ing in frequent:
        targets[ing] = parsed_series.apply(lambda s: int(ing in s)).values

    # --- 4) Обучение моделей ---
    ing_models = {}
    for ing, y in targets.items():
        clf = RidgeClassifier()
        clf.fit(_X, y)
        ing_models[ing] = clf

    return ing_models, frequent




# **This line must run at import-time** so ingredient_models is defined before you use it below:
ingredient_models, frequent_ingredients = train_ingredient_models(food_df, X_combined)


vectorizer_wet, svd_wet, X_text_reduced_wet = build_text_pipeline(food_df[food_df["food_form"]=="wet food"]["description"], n_components=100)
encoder_wet, X_categorical_wet = build_categorical_encoder(food_df[food_df["food_form"]=="wet food"])
X_categorical_wet=apply_category_masks(X_categorical_wet,encoder_wet)
X_combined_wet = combine_features(X_text_reduced_wet, X_categorical_wet)
#X_combined = csr_matrix(X_text_reduced)

@st.cache_resource(show_spinner=False)
def train_nutrient_models(food, _X):
    nutrient_models = {}
    scalers = {}

    nutrients = ['moisture', 'protein', 'fats', 'carbohydrate']
  
    for nutrient in nutrients:
        y = food[nutrient].fillna(food[nutrient].median()).values.reshape(-1, 1)
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
ridge_models, scalers = train_nutrient_models(food_df[food_df["food_form"]=="wet food"], X_combined_wet)

# Кнопки и состояния -----------------------------------------------------------------------------------
# 1 этап выбор характеристик для собаки --------------------------------------------------------------

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
              

if "step" not in st.session_state:
    st.session_state.step = 0  # 0 — начальное, 1 — после генерации, 2 — после расчета

# -----------------------------------
# 8) STREAMLIT UI LAYOUT
# -----------------------------------

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

breed_list = sorted(disease_df["name_breed"].unique())
user_breed = st.selectbox("Порода собаки:", breed_list)

min_weight = disease_df.loc[disease_df["name_breed"] == user_breed, "min_weight"].values
max_weight = disease_df.loc[disease_df["name_breed"] == user_breed, "max_weight"].values
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

#------------------------ выбор функции максимизации и нутриентных ограничен


def extract_target_foods(df, func_name, breed_size, lifestage):
    df_func = df[(df["category"].isin(func_name)) & (df["breed_size"].isin([breed_size, "-"])) & (df["life_stage"].isin([lifestage, "-"]))]
    if len(df_func) == 0:
        df_func = df[(df["category"].isin(func_name)) & (df["life_stage"] == lifestage)]
    if len(df_func) == 0:
        df_func = df[df["category"].isin(func_name)]
    if len(df_func) == 0:
        df_func = df[(df["breed_size"].isin([breed_size, "-"])) & (df["life_stage"].isin([lifestage, "-"]))]
    return df_func

def get_conditions_for_function(df, func_name, breed_size, lifestage):
		df_wet = (food_df[(food_df["food_form"] == "wet food") & (food_df["moisture"] > 50)].copy()).explode("category")
		df_func_w = extract_target_foods(df_wet, func_name, breed_size, lifestage)
		
		df_dry = (food_df[(food_df["food_form"] == "dry food") & (food_df["moisture"] < 50)].copy()).explode("category")
		df_func_dr=extract_target_foods(df_dry, func_name, breed_size, lifestage)		
		
		maximize = [ i for i in main_nutrs  if (df_func_w[i.replace("_per","")].mean() > df_wet[i.replace("_per","")].mean() or df_func_dr[i.replace("_per","")].mean() > df_dry[i.replace("_per","")].mean())]
		return  maximize

#--------------------------------------------------------------------------------------------
# 2 этап настройка условий рецепта  ---------------------------------------------------------

if user_breed:
    info = disease_df[disease_df["name_breed"] == user_breed]
    if not info.empty:
        breed_size = info["breed_size_category"].values[0]
        disorders = info["name_disease"].unique().tolist()+["food sensitivity","weight management"]+[i for i in  ["aging care","puppy care","adult care"] if transl_age[age_type_categ] in i]
        selected_disorder = st.selectbox("Заболевание:", disorders)
        match = info.loc[info["name_disease"] == selected_disorder, "name_disorder"]
        disorder_type = match.iloc[0] if not match.empty else selected_disorder

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
                                                          
            # Build query vector
            keywords = disorder_keywords.get(disorder_type, selected_disorder).lower()
            kw_tfidf = vectorizer.transform([keywords])
            kw_reduced = svd.transform(kw_tfidf)

            # One-hot for (breed_size, age_type_categ)
            cat_vec = encoder.transform([[breed_size, age_type_categ]])
            kw_combined = hstack([csr_matrix(kw_reduced), cat_vec])

            # Rank ingredients
            ing_scores = {
                ing: clf.decision_function(kw_combined)[0]
                for ing, clf in ingredient_models.items()
            }
            top_ings = sorted(ing_scores.items(), key=lambda x: x[1], reverse=True)

            prot=sorted([i for i in top_ings if i[0] in proteins], key=lambda x: x[1], reverse=True)[0][0]
            prot=df_standart[df_standart["name_feed_ingredient"]==prot]["ingredient_full_ru"].tolist()

            carb_cer=sorted([i for i in top_ings if i[0] in carbonates_cer and i[0]!="flaxseed"], key=lambda x: x[1], reverse=True)[0][0]
            carb_cer=df_standart[df_standart["name_feed_ingredient"]==carb_cer]["ingredient_full_ru"].tolist()

            carb_veg=sorted([i for i in top_ings if i[0] in carbonates_veg], key=lambda x: x[1], reverse=True)[0][0]
            carb_veg=df_standart[df_standart["name_feed_ingredient"]==carb_veg]["ingredient_full_ru"].tolist()

            fat=sorted([i for i in top_ings if i[0] in oils], key=lambda x: x[1], reverse=True)[0][0]
            fat=df_standart[df_standart["name_feed_ingredient"]==fat]["ingredient_full_ru"].tolist()
            wat=df_standart[df_standart["name_feed_ingredient"].isin(water)]["ingredient_full_ru"].tolist()
			
            ingredients_finish = [i for i in prot+carb_cer+carb_veg+fat+wat if len(i)>0]

            kw_tfidf = vectorizer_wet.transform([keywords])
            kw_reduced = svd_wet.transform(kw_tfidf)
            cat_vec = encoder_wet.transform([[breed_size, transl_age[age_type_categ]]])
            cat_vec = apply_category_masks(cat_vec,encoder_wet)
            kw_combined = hstack([csr_matrix(kw_reduced), cat_vec])
            nutrient_preds = {}
            for nut, model in ridge_models.items():
                      pred = model.predict(kw_combined)[0]
                      sc = scalers.get(nut)
                      if sc:
                        pred = sc.inverse_transform([[pred]])[0][0]
                      nutrient_preds[nut] = float(round(pred, 2))
			
            # Display
            st.subheader("🌿 Рекомендуемые ингредиенты")
            for ing in ingredients_finish:
                st.write("• " + ing)
            if len(ingredients_finish)>0:               

                      for col in main_nutrs+other_nutrients+major_minerals+vitamins:
                        if col !='ЭПК (50-60%) + ДГК (40-50%), г':
                          ingredirents_df[col] = ingredirents_df[col].astype(str).str.replace(',', '.', regex=False)
                          ingredirents_df[col] = pd.to_numeric(ingredirents_df[col], errors='coerce')
                      ingredirents_df['epa_g(50-60%) + dha_g(40-50%)'] = ingredirents_df['epa_g']*0.5 + ingredirents_df['dha_g']*0.5
                      ingredirents_df[main_nutrs+other_nutrients+major_minerals+vitamins] = ingredirents_df[main_nutrs+other_nutrients+major_minerals+vitamins]
                     
                      proteins=ingredirents_df[ingredirents_df["category_ru"].isin(["Яйца и Молочные продукты", "Мясо"])]["ingredient_format_cat"].tolist()
                      oils=ingredirents_df[ingredirents_df["category_ru"].isin([ "Масло и жир"])]["ingredient_format_cat"].tolist()
                      carbonates_cer=ingredirents_df[ingredirents_df["category_ru"].isin(["Крупы"])]["ingredient_format_cat"].tolist()
                      carbonates_veg=ingredirents_df[ingredirents_df["category_ru"].isin(["Зелень и специи","Овощи и фрукты"])]["ingredient_format_cat"].tolist()
                      other=ingredirents_df[ingredirents_df["category_ru"].isin(["Вода, соль и сахар"])]["ingredient_format_cat"].tolist()

                      meat_len=len(set(proteins).intersection(set(ingredients_finish)))

                      
###################################################################################################################################################################
                
                      if "selected_ingredients" not in st.session_state or st.session_state.show_result_1==False:
                          # Преобразуем ingredients_finish в set и сохраняем
                          st.session_state.selected_ingredients = set(ingredients_finish)

                      st.title("🍲 Выбор ингредиентов")
                      for category in ingredirents_df['category_ru'].dropna().unique():
                          with st.expander(f"{category}"):
                              df_cat = ingredirents_df[ingredirents_df['category_ru'] == category]
                              for ingredient in df_cat['name_ingredient_ru'].dropna().unique():
                                  df_ing = df_cat[df_cat['name_ingredient_ru'] == ingredient]
                                  unique_descs = df_ing['format_ingredient_ru'].dropna().unique()
                                  
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
                                  
                                  # отобразить "обыкновенные" без вложенного expander
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
                      food = ingredirents_df.set_index("ingredient_format_cat")[main_nutrs+other_nutrients+major_minerals+vitamins].to_dict(orient='index')


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
                                ingr_ranges.append(st.slider(f"{ingr.replace(" — Обыкновенный", "")}", 0, 100, (5,35)))

                              elif ingr in carbonates_veg:
                                ingr_ranges.append(st.slider(f"{ingr.replace(" — Обыкновенный", "")}", 0, 100, (5,25)))
                              elif "Вода" in ingr:
                                ingr_ranges.append(st.slider(f"{ingr.replace(" — Обыкновенный", "")}", 0, 100, (0,30)))
                              elif ingr in other:
                                  ingr_ranges.append(st.slider(f"{ingr.replace(" — Обыкновенный", "")}", 0, 100, (1,3)))


                          # --- Ограничения по нутриентам ---
                          st.subheader("Ограничения по нутриентам:")
                          nutr_ranges = {}
                          maximaze_nutrs = get_conditions_for_function(food_df, transl_dis[disorder_type], transl_size[size_categ], transl_age[age_type_categ])
						  
                          needeble_proterin = protein_need_calc(st.session_state.kkal_sel, age_type_categ,  st.session_state.weight_sel, st.session_state.select_reproductive_status, age ,age_metric)					  
                          nutr_ranges['moisture_per'] = st.slider(f"{'Влага'}", 0, 100, (int(nutrient_preds["moisture"]-5), int(nutrient_preds["moisture"]+5)))
                          nutr_ranges['protein_per'] = st.slider(f"{'Белки'}", 0, 100, (int(nutrient_preds["protein"]-3), int(nutrient_preds["protein"]+3)))
                          nutr_ranges['carbohydrate_per'] = st.slider(f"{'Углеводы'}", 0, 100, (int(nutrient_preds["carbohydrate"]-2), int(nutrient_preds["carbohydrate"]+2)))
                          nutr_ranges['fats_per'] = st.slider(f"{'Жиры'}", 0, 100, (int(nutrient_preds["fats"]-1), int(nutrient_preds["fats"]+1)) )
						  
                          if ingr_ranges != st.session_state.prev_ingr_ranges:
                                st.session_state.show_result_2 = False
                                st.session_state.prev_ingr_ranges = ingr_ranges.copy()
                            
                          # Проверяем, изменились ли ограничения по нутриентам
                          if nutr_ranges != st.session_state.prev_nutr_ranges:
                                st.session_state.show_result_2 = False
                                st.session_state.prev_nutr_ranges = nutr_ranges.copy()
                          
                          # --- Построение задачи LP ---
                          A = [
                              [food[ing][nutr]/100 if val > 0 else -food[ing][nutr]/100
                              for ing in ingredient_names]
                              for nutr in nutr_ranges
                              for val in (-nutr_ranges[nutr][0], nutr_ranges[nutr][1])
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
                             [ nutrients_transl.loc[nutrients_transl["name_in_database"] == nutr,"name_ru"].iloc[0].split(",")[0] for nutr in main_nutrs],
                              default=[ nutrients_transl.loc[nutrients_transl["name_in_database"] == nutr,"name_ru"].iloc[0].split(",")[0] for nutr in maximaze_nutrs] 
                          )

                        # Инициализация предыдущего значения
                          if "prev_selected_maximize" not in st.session_state:
                            st.session_state.prev_selected_maximize = [nutrients_transl.loc[nutrients_transl["name_in_database"] == nutr,"name_ru"].iloc[0].split(",")[0] for nutr in main_nutrs]
                        
                        # Проверка изменений
                          if selected_maximize != st.session_state.prev_selected_maximize:
                            st.session_state.show_result_2 = False
                            st.session_state.prev_selected_maximize = selected_maximize.copy()
                          selected_maximize=[nutrients_transl.loc[nutrients_transl["name_ru"].str.contains(nutr, na=False),"name_in_database"].iloc[0] for nutr in selected_maximize]
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
                                      st.write(f"{name.replace(" — Обыкновенный", "")}: **{int(round(value,0))} г**")

                                  st.markdown("### 💪 Питательная ценность на 100 г:")
                                  nutrients = {
                                      nutr: round(sum(res.x[i] * food[name][nutr]/100 for i, name in enumerate(ingredient_names)) * 100, 2)
                                      for nutr in main_nutrs
                                  }
                                  for k, v in nutrients.items():
                                      k_trl = nutrients_transl.loc[nutrients_transl["name_in_database"] == k,"name_ru"].iloc[0].split(",")[0]
                                      st.write(f"**{k_trl}:** {int(round(v,0))} г")
                                  en_nutr_100=3.5*nutrients["protein_per"]+8.5*nutrients["fats_per"]+3.5*nutrients["carbohydrate_per"]
                                  st.write(f"**Энергетическая ценность:** {int(round(en_nutr_100,0))} ккал")

                                  st.write(f"****")

                            
                                  st.markdown(f"### Сколько нужно в граммах корма и ингредиентов на {int(round(metobolic_energy,0))} ккал")           
                                  needed_feed_g = (metobolic_energy * 100) / en_nutr_100
                                  ingredients_required = {
                                      name: round((weight * needed_feed_g / 100), 2)
                                      for name, weight in result.items()
                                  }                                  
                                  st.write(f"📌 Корм: {int(round(needed_feed_g, 0))} г")
                                  st.write("🧾 Количество ингредиентов для этой порции:")
                                  for ingredient, amount in ingredients_required.items():
                                      st.write(f" - {ingredient.replace(" — Обыкновенный", "")}: {int(round(amount,0))} г")

                                
                                  count_nutr_cont_all = {
                                      nutr: round(sum(amount * food[ingredient][nutr]/100 for ingredient, amount in ingredients_required.items()), 2)
                                      for nutr in main_nutrs+other_nutrients+major_minerals+vitamins
                                  }

                                  st.markdown(f"### 💪 Питательная ценность на {int(round(needed_feed_g, 0))} г:")

                                  for k in main_nutrs:
                                      k_trl=nutrients_transl.loc[nutrients_transl["name_in_database"] == k,"name_ru"].iloc[0].split(",")[0]
                                      st.write(f"**{k_trl}:** {int(round(count_nutr_cont_all[k], 0))} г")
                                  st.write(f"****") 
                                
                                  show_nutr_content(count_nutr_cont_all, other_nutrient_norms,nutrients_transl)    
                                
                            
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
                            
                                            totals = {nutr: 0.0 for nutr in main_nutrs}
                                            for i, ingr in enumerate(ingredient_names):
                                                for nutr in main_nutrs:
                                                    totals[nutr] += values[ingr] * food[ingr][nutr]/100
                            
                                            # Штраф за отклонения от допустимых диапазонов
                                            penalty = 0
                                            for nutr in main_nutrs:
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
                                        st.write(f"{name.replace(" — Обыкновенный", "")}: **{int(round(val, 0))} г**")
 
                                    
                                    st.markdown("### 💪 Питательная ценность на 100 г:")
                                    for nutr in main_nutrs:
                                        nutr_trl=nutrients_transl.loc[nutrients_transl["name_in_database"] == nutr,"name_ru"].iloc[0].split(",")[0]
                                        st.write(f"**{nutr_trl}:** {int(round(totals[nutr], 0))} г")
										
                                    en_nutr_100=3.5*totals["protein_per"]+8.5*totals["fats_per"]+3.5*totals["carbohydrate_per"]
                                    st.write(f"**Энергетическая ценность:** {int(round(en_nutr_100,0))} ккал")



                                    
                                    st.markdown(f"### Сколько нужно в граммах корма и ингредиентов на {int(round(metobolic_energy,0))} ккал")           
                                    needed_feed_g = (metobolic_energy * 100) / en_nutr_100
                                    ingredients_required = {
                                        name: round((weight * needed_feed_g / 100), 2)
                                        for name, weight in values.items()
                                    }                                  
                                    st.write(f"📌 Корм: {round(needed_feed_g, 2)} г")
                                    st.write("🧾 Количество ингредиентов для этой порции:")
                                    for ingredient, amount in ingredients_required.items():
                                        st.write(f" - {ingredient.replace(" — Обыкновенный", "")}: {int(round(amount,0))} г")

                                    count_nutr_cont_all = {
                                      nutr: round(sum(amount * food[ingredient][nutr]/100 for ingredient, amount in ingredients_required.items()), 2)
                                      for nutr in main_nutrs+other_nutrients+major_minerals+vitamins }
                                    

                                    st.markdown(f"### 💪 Питательная ценность на {int(round(needed_feed_g, 0))} г:")

                                    for k in main_nutrs:
                                      k_trl=nutrients_transl.loc[nutrients_transl["name_in_database"] == k,"name_ru"].iloc[0].split(",")[0]
                                      st.write(f"**{k_trl}:** {int(round(count_nutr_cont_all[k],0))} г")
                                    st.write(f"****") 
                                    show_nutr_content(count_nutr_cont_all, other_nutrient_norms,nutrients_transl)   






                                    
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
                                    ax2.set_xticklabels([nutrients_transl.loc[nutrients_transl["name_in_database"] == nutr,"name_ru"].iloc[0].split(",")[0] for nutr in  nutrients], rotation=0)
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
