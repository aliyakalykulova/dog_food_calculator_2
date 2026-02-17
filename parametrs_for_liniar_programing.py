
def ingredients_choose():
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
  
    if "selected_ingredients" not in st.session_state or st.session_state.show_result_1==False:
         st.session_state.selected_ingredients = set(ingredients_finish)

    st.title("🍲 Выбор ингредиентов")
       for category in ['category_ru'].dropna().unique():
           with st.expander(f"{category}"):
                df_cat = ingredirents_df[ingredirents_df['category_ru'] == category]
                for ingredient in df_cat['name_ingredient_ru'].dropna().unique():
                    df_ing = df_cat[df_cat['name_ingredient_ru'] == ingredient]
                    unique_descs = df_ing['format_ingredient_ru'].dropna().unique()
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

def nutrients_limits():
        st.subheader("Ограничения по нутриентам:")
        nutr_ranges = {}
        maximaze_nutrs = get_conditions_for_function(food_df, disorder_type, breed_size, age_type_categ)
						  
        needeble_proterin = protein_need_calc( age_type_categ)					  
        nutr_ranges['moisture_per'] = st.slider(f"{'Влага'}", 0, 100, (int(nutrient_preds["moisture"]-5), int(nutrient_preds["moisture"]+5)))
        nutr_ranges['protein_per'] = st.slider(f"{'Белки'}", 0, 100, (int(nutrient_preds["protein"]-3), int(nutrient_preds["protein"]+3)))
        nutr_ranges['carbohydrate_per'] = st.slider(f"{'Углеводы'}", 0, 100, (int(nutrient_preds["carbohydrate"]-2), int(nutrient_preds["carbohydrate"]+2)))
        nutr_ranges['fats_per'] = st.slider(f"{'Жиры'}", 0, 100, (int(nutrient_preds["fats"]-1), int(nutrient_preds["fats"]+1)) )
						  
def ingredients_limits():
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


               

def lin_prog_parametrs():
  
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


def parametrs_for_liniar_programing():
    ingredients_choose()
    ingredients_limits()
    nutrients_limits()
    if ingr_ranges != st.session_state.prev_ingr_ranges:
       st.session_state.show_result_2 = False
       st.session_state.prev_ingr_ranges = ingr_ranges.copy()
    if nutr_ranges != st.session_state.prev_nutr_ranges:
       st.session_state.show_result_2 = False
       st.session_state.prev_nutr_ranges = nutr_ranges.copy()


