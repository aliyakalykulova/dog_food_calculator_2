import streamlit as st
import pandas as pd
from ctypes import create_string_buffer
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


def show_resuts_success(ingredient_names,res,food,main_nutrs,nutrients_transl,metobolic_energy,other_nutrients,major_minerals,vitamins,
                        count_nutr_cont_all,kkal_sel, age_type_categ, weight_sel, select_reproductive_status):
         st.success("✅ Решение найдено!")
         result = {name: round(val * 100, 2) for name, val in zip(ingredient_names, res.x)}
         st.markdown("### 📦 Состав (в граммах на 100 г):")
         for name, value in result.items():
              st.write(f"{name.replace(" — Обыкновенный", "")}: **{int(round(value,0))} г**")
         st.markdown("### 💪 Питательная ценность на 100 г:")
         nutrients = {nutr: round(sum(res.x[i] * food[name][nutr]/100 for i, name in enumerate(ingredient_names)) * 100, 2)
                      for nutr in main_nutrs}
         for k, v in nutrients.items():
              k_trl = nutrients_transl.loc[nutrients_transl["name_in_database"] == k,"name_ru"].iloc[0].split(",")[0]
              st.write(f"**{k_trl}:** {int(round(v,0))} г")
         en_nutr_100=3.5*nutrients["protein_per"]+8.5*nutrients["fats_per"]+3.5*nutrients["carbohydrate_per"]
         st.write(f"**Энергетическая ценность:** {int(round(en_nutr_100,0))} ккал")
         st.write(f"****")                      
         st.markdown(f"### Сколько нужно в граммах корма и ингредиентов на {int(round(metobolic_energy,0))} ккал")           
         needed_feed_g = (metobolic_energy * 100) / en_nutr_100
         ingredients_required = { name: round((weight * needed_feed_g / 100), 2)
                                  for name, weight in result.items() }               
         st.write(f"📌 Корм: {int(round(needed_feed_g, 0))} г")
         st.write("🧾 Количество ингредиентов для этой порции:")
         for ingredient, amount in ingredients_required.items():
              st.write(f" - {ingredient.replace(" — Обыкновенный", "")}: {int(round(amount,0))} г")
         count_nutr_cont_all = {nutr: round(sum(amount * food[ingredient][nutr]/100 for ingredient, amount in ingredients_required.items()), 2)
                                for nutr in main_nutrs+other_nutrients+major_minerals+vitamins}

         st.markdown(f"### 💪 Питательная ценность на {int(round(needed_feed_g, 0))} г:")
         for k in main_nutrs:
              k_trl=nutrients_transl.loc[nutrients_transl["name_in_database"] == k,"name_ru"].iloc[0].split(",")[0]
              st.write(f"**{k_trl}:** {int(round(count_nutr_cont_all[k], 0))} г")
         st.write(f"****") 
         show_nutr_content(count_nutr_cont_all,nutrients_transl, kkal_sel, age_type_categ, weight_sel, select_reproductive_status)    

def show_figures_ingr_nutr(ingredient_names,ingr_ranges,nutr_ranges,totals,nutrients_transl):
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


def show_resuts_success_2(best_recipe,main_nutrs,metobolic_energy,food,other_nutrients,major_minerals,vitamins,count_nutr_cont_all,ingredient_names,
                          ingr_ranges,nutr_ranges,nutrients_transl, kkal_sel, age_type_categ, weight_sel, select_reproductive_status):
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
     ingredients_required = {name: round((weight * needed_feed_g / 100), 2)
                             for name, weight in values.items()}                                  
     st.write(f"📌 Корм: {round(needed_feed_g, 2)} г")
     st.write("🧾 Количество ингредиентов для этой порции:")
     for ingredient, amount in ingredients_required.items():
            st.write(f" - {ingredient.replace(" — Обыкновенный", "")}: {int(round(amount,0))} г")
     count_nutr_cont_all = {nutr: round(sum(amount * food[ingredient][nutr]/100 for ingredient, amount in ingredients_required.items()), 2)
                            for nutr in main_nutrs+other_nutrients+major_minerals+vitamins }
     st.markdown(f"### 💪 Питательная ценность на {int(round(needed_feed_g, 0))} г:")
  
     for k in main_nutrs:
         k_trl=nutrients_transl.loc[nutrients_transl["name_in_database"] == k,"name_ru"].iloc[0].split(",")[0]
         st.write(f"**{k_trl}:** {int(round(count_nutr_cont_all[k],0))} г")
     st.write(f"****") 
     show_nutr_content(count_nutr_cont_all,nutrients_transl, kkal_sel, age_type_categ, weight_sel, select_reproductive_status)   
     show_figures_ingr_nutr(ingredient_names,ingr_ranges,nutr_ranges,totals,nutrients_transl)

                                    
                                    

def calc_recipe(ingr_ranges,ingredient_names,main_nutrs,food,nutr_ranges):
   st.error("❌ Не удалось найти оптимальное решение. Попробуйте другие параметры.")
   with st.spinner("🔄 Ищем по другому методу..."):
   step = 1  # шаг в процентах
   variants = []
   ranges = [np.arange(low, high + step, step) for (low, high) in ingr_ranges]
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
   return best_recipe

