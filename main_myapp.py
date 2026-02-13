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

from kcal_calculate import kcal_calculate
from kcal_calculate import size_category
from kcal_calculate import age_type_category
from kcal_calculate import bar_print
from kcal_calculate import show_nutr_content
from kcal_calculate import protein_need_calc
from scipy.sparse import csr_matrix
from kcal_calculate import get_conditions_for_function
from kcal_calculate import  apply_category_masks

from kcal_calculate import  load_data
from kcal_calculate import build_text_pipeline
from kcal_calculate import build_categorical_encoder
from kcal_calculate import combine_features
from kcal_calculate import train_ingredient_models
from kcal_calculate import train_nutrient_models
from kcal_calculate import show_sidebar
from show_dog_charecteristics import show_dog_characterictics
from show_ingredients import ingredient_recomendation
from show_ingredients import nutrients_recomendation
from show_results import show_resuts_success
from show_results import show_figures_ingr_nutr
from show_results import show_resuts_success_2
from show_results import calc_recipe

# все спсики-------------------------------------------------------------------------
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
vitamins=['vitamin_a_mcg', 'vitamin_e_mg', 'vitamin_d_mcg', 'vitamin_b1_mg', 'vitamin_b2_mg', 'vitamin_b3_mg', 'vitamin_b5_mg', 'vitamin_b6_mg', 'vitamin_b9_mcg', 'vitamin_b12_mcg', 'vitamin_c_mg', 'vitamin_k_mcg']


food_df, disease_df, df_standart, ingredirents_df,nutrients_transl= load_data()

vectorizer, svd, X_text_reduced = build_text_pipeline(food_df["description"], n_components=100)
encoder, X_categorical = build_categorical_encoder(food_df)
X_categorical=apply_category_masks(X_categorical,encoder)
X_combined = combine_features(X_text_reduced, X_categorical)

ingredient_models, frequent_ingredients = train_ingredient_models(food_df, X_combined)
vectorizer_wet, svd_wet, X_text_reduced_wet = build_text_pipeline(food_df[food_df["food_form"]=="wet food"]["description"], n_components=100)
encoder_wet, X_categorical_wet = build_categorical_encoder(food_df[food_df["food_form"]=="wet food"])
X_categorical_wet=apply_category_masks(X_categorical_wet,encoder_wet)
X_combined_wet = combine_features(X_text_reduced_wet, X_categorical_wet)

ridge_models, scalers = train_nutrient_models(food_df[food_df["food_form"]=="wet food"], X_combined_wet)

# Кнопки и состояния -----------------------------------------------------------------------------------
# 1 этап выбор характеристик для собаки --------------------------------------------------------------

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

if "step" not in st.session_state:
    st.session_state.step = 0  # 0 — начальное, 1 — после генерации, 2 — после расчета

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
	
user_breed, breed_size, avg_wight, age_type_categ = show_dog_characterictics(disease_df)


#--------------------------------------------------------------------------------------------
# 2 этап настройка условий рецепта  ---------------------------------------------------------

if user_breed:
    info = disease_df[disease_df["name_breed"] == user_breed]
    if not info.empty:
        disorders = info["name_disease"].unique().tolist()+["food sensitivity","weight management"]+[i for i in  ["aging care","puppy care","adult care"] if age_type_categ in i]
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
            kcal =kcal_calculate(st.session_state.select_reproductive_status, st.session_state.show_res_berem_time, st.session_state.show_res_num_pup ,  st.session_state.show_res_lact_time, 
                                age_type_categ, st.session_state.weight_sel, avg_wight,  st.session_state.activity_level_sel, user_breed, age)
            metobolic_energy = st.number_input("Киллокаллории в день", min_value=0.0, step=0.1,  value=round(kcal,1) )
			
            if st.session_state.kkal_sel!=metobolic_energy:
               st.session_state.kkal_sel=metobolic_energy
               st.session_state.show_result_1 = True
               st.session_state.show_result_2 = False

            ingredients_finish=ingredient_recomendation(ingredient_models,breed_size, age_type_categ,disorder_type, selected_disorder,vectorizer,svd,encoder, df_standart)
            nutrient_preds = nutrients_recomendation(vectorizer_wet,keywords,svd_wet,encoder_wet, breed_size, age_type_categ, ridge_models,scalers )
  
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
                          maximaze_nutrs = get_conditions_for_function(food_df, disorder_type, breed_size, age_type_categ)
						  
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
                                   show_resuts_success(ingredient_names,res,food,main_nutrs,nutrients_transl,metobolic_energy,other_nutrients,major_minerals,vitamins,
                                   count_nutr_cont_all,st.session_state.kkal_sel, age_type_categ, st.session_state.weight_sel, st.session_state.select_reproductive_status)                            
                              else:
                                  best_recipe=calc_recipe(ingr_ranges,ingredient_names,main_nutrs,food,nutr_ranges)
                                  if best_recipe:
                                           show_resuts_success_2(best_recipe,main_nutrs,metobolic_energy,food,other_nutrients,major_minerals,vitamins,count_nutr_cont_all,ingredient_names,
                                                           ingr_ranges,nutr_ranges,nutrients_transl, st.session_state.kkal_sel, age_type_categ, st.session_state.weight_sel, 
														   st.session_state.select_reproductive_status)            
                                  else:
                                     st.error("🚫 Не удалось найти подходящий состав даже вручную.")
                      else:
                          st.info("👈 Пожалуйста, выберите хотя бы один ингредиент.")
