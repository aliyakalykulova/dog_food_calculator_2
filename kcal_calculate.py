import textwrap
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import sqlite3
import pandas as pd
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
import sqlite3


metrics_age_types=["в годах","в месецах"]
gender_types=["Самец", "Самка"]
rep_status_types=["Нет", "Щенность (беременность)", "Период лактации"]
berem_time_types=["первые 4 недедели беременности","последние 5 недель беременности"]
lact_time_types=["1 неделя","2 неделя","3 неделя","4 неделя"]
age_category_types=["puppy","adult","senior"]
size_types=["small",  "medium",  "large"]

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
vitamins=['vitamin_a_mcg', 'vitamin_e_mg', 'vitamin_d_mcg', 'vitamin_b1_mg', 'vitamin_b2_mg', 'vitamin_b3_mg', 'vitamin_b5_mg', 'vitamin_b6_mg', 'vitamin_b9_mcg', 'vitamin_b12_mcg']



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

@st.cache_resource(show_spinner=False)
def build_text_pipeline(corpus, n_components=100):
    vect = TfidfVectorizer(stop_words="english", max_features=5000)
    X_tfidf = vect.fit_transform(corpus)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    return vect, svd, X_reduced



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


# -----------------------------------
# 6) COMBINE FEATURES INTO SPARSE MATRIX
# -----------------------------------

@st.cache_resource(show_spinner=False)
def combine_features(text_reduced, _cat_matrix):
    # Turn dense text_reduced into sparse form
    X_sparse_text = csr_matrix(text_reduced)
    return hstack([X_sparse_text, _cat_matrix])


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

def get_conditions_for_function(food_df, func_name, breed_size, lifestage):
		df_wet = (food_df[(food_df["food_form"] == "wet food") & (food_df["moisture"] > 50)].copy()).explode("category")
		df_func_w = extract_target_foods(df_wet, func_name, breed_size, lifestage)
		
		df_dry = (food_df[(food_df["food_form"] == "dry food") & (food_df["moisture"] < 50)].copy()).explode("category")
		df_func_dr=extract_target_foods(df_dry, func_name, breed_size, lifestage)		
		
		maximize = [ i for i in main_nutrs  if (df_func_w[i.replace("_per","")].mean() > df_wet[i.replace("_per","")].mean() or df_func_dr[i.replace("_per","")].mean() > df_dry[i.replace("_per","")].mean())]
		return  maximize



def protein_need_calc(kkal, age_type_categ,  w, reproductive_status, age, age_mesuare_type):
   protein_n=0
   if age_type_categ==age_category_types[0]:
         protein_n = 56.3 * kkal / 1000 if (age_mesuare_type == metrics_age_types[1] and age <= 3) else 43.8 * kkal / 1000
   elif reproductive_status==rep_status_types[1] or reproductive_status==rep_status_types[2]:
         protein_n=  50*kkal/1000
   else:
         protein_n=  3.28*(w**0.75)
   return protein_n



def show_nutr_content(count_nutr_cont_all, other_nutrient_norms, nutrients_transl):
                                  for i in range(0, len(other_nutrients_1), 2):
                                      cols = st.columns(2)
                                      for j, col in enumerate(cols):
                                          if i + j < len(other_nutrients_1):
                                              nutris = (other_nutrients_1)[i + j]
                                              nutr_text=nutrients_transl.loc[nutrients_transl["name_in_database"] == nutris,"name_ru"].iloc[0].split(",")    
                                              emg=""
                                              if len(nutr_text)>1:
                                                if "%" not in nutr_text[-1]:
                                                    emg=nutr_text[-1].strip()
                                                else:
                                                  emg="g"
                                              else:
                                                emg="g"
                                              with col:
                                                  st.markdown(f"**{nutr_text[0]}**: {count_nutr_cont_all.get(nutris, '')} {emg}")

                                  coli, colii=st.columns([6,3])
                                  with coli:
                                     for i in range(0, len(other_nutrients_2)):
                                              nutris = other_nutrients_2[i]
                                              nutr_text=nutrients_transl.loc[nutrients_transl["name_in_database"] == nutris,"name_ru"].iloc[0].split(",")    
                                              emg = nutr_text[-1].strip() if len(nutr_text)>1 and "%" not in nutr_text[-1] else "g"
                                              if nutr_text[0] in other_nutrient_norms:
                                                norma = other_nutrient_norms[nutr_text[0]]
                                                st.pyplot(bar_print(norma, count_nutr_cont_all.get(nutris, ''), nutr_text[0]+", "+ emg, str(emg)))
                               
                                  st.markdown("#### 🔹 Минералы")
                                  coli, colii=st.columns([6,3])
                                  with coli:
                                     for i in range(0, len(major_minerals)):
                                              nutris = major_minerals[i]
                                              nutr_text=nutrients_transl.loc[nutrients_transl["name_in_database"] == nutris,"name_ru"].iloc[0].split(",") 
                                              emg = nutr_text[-1].strip() if len(nutr_text)>1 and "%" not in nutr_text[-1] else "g"
                                              norma = other_nutrient_norms[nutris]
                                              st.pyplot(bar_print(norma, count_nutr_cont_all.get(nutris, ''), nutr_text[0]+", "+ emg, str(emg)))
                                                  
                                  st.markdown("#### 🍊 Витамины")
                                  coli, colii=st.columns([6,3])
                                  with coli:
                                     for i in range(0, len(vitamins)):
                                              nutris = vitamins[i]
                                              nutr_text=nutrients_transl.loc[nutrients_transl["name_in_database"] == nutris,"name_ru"].iloc[0].split(",") 
                                              emg = nutr_text[-1].strip() if len(nutr_text)>1 and "%" not in nutr_text[-1] else "g"
                                              norma = other_nutrient_norms[nutris]
                                              st.pyplot(bar_print(norma, count_nutr_cont_all.get(nutris, ''), nutr_text[0]+", "+ emg, str(emg)))

                                  st.markdown("### Необходимо добавить")
                                  for name,amount in count_nutr_cont_all.items():
                                    if name in other_nutrient_norms:
                                      diff=other_nutrient_norms[name] - amount
                                      if diff>0:
                                         name_n=nutrients_transl.loc[nutrients_transl["name_in_database"] == name,"name_ru"].iloc[0].split(",") 
                                         emg = name_n[-1].strip() if len(name_n)>1 and "%" not in name_n[-1] else "g"
                                         st.write(f"**{name_n[0]}:** {round(diff,2)} {emg}")
                                        


def get_other_nutrient_norms(kkal, age_type_categ,  w, reproductive_status):
   if age_type_categ==age_category_types[0]:
         nutrients_per_1000_kcal = {
              "calcium_mg": 3000*kkal/1000,
              "phosphorus_mg": 2500*kkal/1000,
              "magnesium_mg": 100*kkal/1000,
              "sodium_mg": 550*kkal/1000,
              "potassium_mg": 1100*kkal/1000,
              "iron_mg": 22*kkal/1000,
              "copper_mg": 2.7*kkal/1000,
              "zinc_mg": 25*kkal/1000,
              "manganese_mg": 1.4*kkal/1000,

              "vitamin_a_mcg": 378.9*kkal/1000,
              "vitamin_d_mcg": 3.4*kkal/1000,
              "vitamin_e_mg": 7.5*kkal/1000,
              "vitamin_b1_mg": 0.34*kkal/1000,
              "vitamin_b2_mg": 1.32*kkal/1000,
              "vitamin_b3_mg": 4.25*kkal/1000,
              "vitamin_b6_mg": 0.375*kkal/1000,
              "vitamin_b12_mcg": 8.75*kkal/1000,
                         
              "selenium_mcg": 87.5*kkal/1000,
              "choline_mg": 425*kkal/1000,
              "vitamin_b5_mg": 3.75*kkal/1000,
              "linoleic_acid_g": 3.3*kkal/1000,
              "vitamin_b9_mcg": 68*kkal/1000,
              "alpha_linolenic_acid_g": 0.2*kkal/1000,
              "arachidonic_acid_g": 0.08*kkal/1000,
              "epa_g(50-60%) + dha_g(40-50%)": 0.13*kkal/1000,
           
              "iodine_mcg": 220*kkal/1000,
              "Биотин (мкг)": 4*kkal/1000
             }

         return nutrients_per_1000_kcal
     
   elif reproductive_status==rep_status_types[1] or reproductive_status==rep_status_types[2]:
         nutrients_per_1000_kcal = {
    "calcium_mg": 1900*kkal/1000,
    "phosphorus_mg": 1200*kkal/1000,
    "magnesium_mg": 150*kkal/1000,
    "sodium_mg": 500*kkal/1000,
    "potassium_mg": 900*kkal/1000,
    "iron_mg": 17*kkal/1000,
    "copper_mg": 3.1*kkal/1000,
    "zinc_mg": 24*kkal/1000,
    "manganese_mg": 1.8*kkal/1000,

    "vitamin_a_mcg": 378.9*kkal/1000,
    "vitamin_d_mcg": 3.4*kkal/1000,
    "vitamin_e_mg": 7.5*kkal/1000,
    "vitamin_b1_mg": 0.56*kkal/1000,
    "vitamin_b2_mg": 1.3*kkal/1000,
    "vitamin_b3_mg": 4.25*kkal/1000,
    "vitamin_b6_mg": 0.375*kkal/1000,
    "vitamin_b12_mcg": 8.75*kkal/1000,

    "selenium_mcg": 87.5*kkal/1000,
    "choline_mg": 425*kkal/1000,
    "vitamin_b5_mg": 3.75*kkal/1000,
    "vitamin_b9_mcg": 67.5*kkal/1000,
    "linoleic_acid_g": 3.3*kkal/1000,
    "alpha_linolenic_acid_g": 0.2*kkal/1000,
    "epa_g(50-60%) + dha_g(40-50%)": 0.13*kkal/1000,

    "iodine_mcg": 220*kkal/1000,
    "Биотин": 4*kkal/1000
         }
         return nutrients_per_1000_kcal

   else:  
      other_for_adult = {
    "calcium_mg": 130*(w**0.75),
    "phosphorus_mg": 100*(w**0.75),
    "magnesium_mg": 19.7*(w**0.75),
    "sodium_mg": 26.2*(w**0.75),
    "potassium_mg": 140*(w**0.75),
    "iron_mg": 1.0*(w**0.75),
    "copper_mg": 0.2*(w**0.75),
    "zinc_mg": 2.0*(w**0.75),
    "manganese_mg": 0.16*(w**0.75),

    "vitamin_a_mcg": 4.175*(w**0.75),
    "vitamin_d_mcg": 0.45*(w**0.75),
    "vitamin_e_mg": 1.0*(w**0.75),
    "vitamin_b1_mg": 0.074*(w**0.75),
    "vitamin_b2_mg": 0.171*(w**0.75),
    "vitamin_b3_mg": 0.57*(w**0.75),
    "vitamin_b6_mg": 0.049*(w**0.75),
    "vitamin_b12_mcg": 1.15*(w**0.75),

    "selenium_mcg": 11.8*(w**0.75),
    "iodine_mcg": 29.6*(w**0.75),
    "vitamin_b5_mg": 0.49*(w**0.75),
    "vitamin_b9_mcg": 8.9*(w**0.75),
    "choline_mg": 56*(w**0.75),
    "linoleic_acid_g": 0.36*(w**0.75),
    "alpha_linolenic_acid_g": 0.014*(w**0.75),
    "epa_g(50-60%) + dha_g(40-50%)": 0.03*(w**0.75)
       }
      return other_for_adult



nutrients_per_kg = {
    "Витамин А (МЕ/кг)": 34000,
    "Витамин D3 (МЕ/кг)": 1100,
    "Витамин Е (МЕ/кг)": 350,
    "Железо (мг/кг)": 120,
    "Йод (мг/кг)": 1.9,
    "Медь (мг/кг)": 13,
    "Марганец (мг/кг)": 46,
    "Цинк (мг/кг)": 110,
    "Селен (мг/кг)": 0.13
}





def bar_print(total_norm,current_value,name_ing,mg):
                                        maxi_dat = total_norm if total_norm>current_value else current_value
                                        norma = 100 if maxi_dat== total_norm else (total_norm/current_value)*100
                                        curr =  100 if maxi_dat== current_value else (current_value/total_norm)*100
  
                                        maxi_lin = 100*1.2
                                        diff = current_value - total_norm
                                        fig, ax = plt.subplots(figsize=(5, 1))
                                        ax.axis('off')
                                        # Добавляем запас 20% справа и фиксируем начало оси X
                                        ax.set_xlim(-60, maxi_lin+8)
                                        ax.set_ylim(-0.5, 0.5)
                                        ax.plot([0, maxi_lin], [0, 0], color='#e0e0e0', linewidth=10, solid_capstyle='round', alpha=0.8)
                                        fixed_space = -10 
                                        wrapped_text = "\n".join(textwrap.wrap(name_ing, width=15))
                                        ax.text(fixed_space, 0, wrapped_text, ha='right', va='center', fontsize=13)
                                        if current_value < total_norm:
                                            ax.plot([0, norma], [0, 0], color='green', linewidth=10, solid_capstyle='round')
                                            ax.plot([0, curr], [0, 0], color='purple', linewidth=10, solid_capstyle='round')
                                        else:
                                            ax.plot([0, curr], [0, 0], color='darkgray', linewidth=10, solid_capstyle='round')
                                            ax.plot([0, norma], [0, 0], color='green', linewidth=10, solid_capstyle='round')
                                        if diff < 0:
                                              ax.text(maxi_lin+10, 0,
                                                f"Дефицит: {round(abs(diff),2)} {mg}",
                                                ha='left', va='center', fontsize=13, color='black')
                                        else:
                                               ax.text(maxi_lin+10, 0,"                            ", ha='left', va='center', fontsize=13, color='black')
                                        ax.text(curr, 0.2, f"Текущее\n{round(current_value,2)}", color='purple', ha='center', va='bottom', fontsize=9)
                                        ax.text(norma, -0.2,  f"Норма\n{round(total_norm,2)}", color='green', ha='center', va='top', fontsize=9)
                                        return fig

def size_category(df):
    w = (df["min_weight"].iloc[0] + df["max_weight"].iloc[0]) / 2  
    if w <= 10:
        return size_types[0],w
    elif w <= 25:
        return size_types[1],w
    else :
        return size_types[2],w

def age_type_category(size_categ, age ,age_metric):
        if age_metric==metrics_age_types[0]:
            age=age*12
            
        if size_categ==size_types[0]:
          if age>=1*12 and age<=8*12:    
             return age_category_types[1]
          elif age<1*12:    
             return age_category_types[0]
          elif age>8*12:  
             return age_category_types[2]
       
        elif size_categ==size_types[2]:
          if age>=15 and age<=7*12  :   
              return age_category_types[1]
          elif age<15:     
             return age_category_types[0]
          elif age>7*12:    
             return age_category_types[2]
              
        else:
          if age<=6*12 and age>=24:    
              return age_category_types[1]
          elif age<24:    
              return age_category_types[0]
          elif age>6*12:   
              return age_category_types[2]
            
def kcal_calculate(reproductive_status, berem_time, num_pup, L_time, age_type, weight, expected, activity_level, user_breed, age):
    formula=""
    page=""
    if L_time==lact_time_types[0]:
      L=0.75
    elif L_time==lact_time_types[1]:
      L=0.95
    elif L_time==lact_time_types[2]:
      L=1.1
    else :
      L=1.2
    
    if reproductive_status==rep_status_types[1]:
      if berem_time==berem_time_types[0]:
        kcal=132*(weight**0.75)
        formula= r"kcal = 132 \cdot вес^{0.75}  \\  \text{(первые 4 недели беременности)}"
        page = "56"
        
      else:
        kcal=132*(weight**0.75) + (26*weight)
        formula= r"kcal = 132 \cdot вес^{0.75} + 26 \cdot вес  \\  \text{(последние 5 недель беременности)}"
        page = "56"
  
    elif reproductive_status==rep_status_types[2]:
       if num_pup<5:
         kcal=145*(weight**0.75) + 24*num_pup*weight*L
         formula = fr"kcal = 145 \cdot вес^{{0.75}} + 24 \cdot n \cdot вес \cdot L  \\  \text{{n - количество щенков}}  \\  \text{{L = {L} для {L_time}}}"
         page = "56"
         
       else:
         kcal=145*(weight**0.75) + (96+12*num_pup-4)*weight*L
         formula = fr"kcal = 145 \cdot вес^{{0.75}}  + (96 + 12 \cdot n - 4) \cdot вес \cdot L    \\  \text{{n - количество щенков}}  \\  \text{{L = {L} для {L_time}}}"       
         page = "56"
         
    else:
      if age_type==age_category_types[0]:
          if age<2:
            kcal=25 * weight 
            formula= r"kcal = 25 \cdot вес"
            page = "56"
            
          elif age>=2 and age <12:
            kcal=(254.1-135*(weight/expected) )*(weight**0.75)
            formula=fr"kcal = \left(254.1 - 135 \cdot \frac{{вес}}{{w}}\right) \cdot вес^{{0.75}}  \\  w = {round(expected,2)}  \text{{кг ;  предположительный вес для породы {user_breed}}}"
            page = "56"

        
          else :
            kcal=130*(weight**0.75)
            formula= r"kcal = 130 \cdot вес^{0.75}"
            page = "54"


      
      elif age_type==age_category_types[2]:
          if activity_level==activity_level_cat_2[0]:
              kcal=80*(weight**0.75)
              formula= r"kcal = 80  \cdot вес^{0.75}"
              page = "54"
        
          elif activity_level==activity_level_cat_2[1]:
              kcal=95*(weight**0.75)
              formula= r"kcal = 95  \cdot вес^{0.75}"
              page = "54"    
            
          else:
             kcal=110*(weight**0.75)
             formula= r"kcal = 110  \cdot вес^{0.75}"
             page = "54"
            
      else:   
            if activity_level==activity_level_cat_1[0]:
              kcal=95*(weight**0.75)
              formula= r"kcal = 95  \cdot вес^{0.75}"
              page = "55"
        
            elif activity_level==activity_level_cat_1[1]:
              kcal=110*(weight**0.75)
              formula= r"kcal = 110  \cdot вес^{0.75}"
              page = "55"
        
            elif activity_level==activity_level_cat_1[2]:
              kcal=125*(weight**0.75)
              formula= r"kcal = 125  \cdot вес^{0.75}"
              page = "55"
        
            elif activity_level==activity_level_cat_1[3]:
              kcal=160*(weight**0.75)
              formula= r"kcal = 160  \cdot вес^{0.75}"
              page = "55"
              
            elif activity_level==activity_level_cat_1[4]:
              kcal=860*(weight**0.75)
              formula= r"kcal = 860  \cdot вес^{0.75}"
              page = "55"
           
            else:
              kcal=90*(weight**0.75)
              formula= r"kcal = 90  \cdot вес^{0.75}"
              page = "55"
              
    return kcal, formula, page
