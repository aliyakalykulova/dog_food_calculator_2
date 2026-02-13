import streamlit as st
from kcal_calculate import show_sidebar
from kcal_calculate import size_category
from kcal_calculate import age_type_category

metrics_age_types=["в годах","в месецах"]
gender_types=["Самец", "Самка"]
rep_status_types=["Нет", "Щенность (беременность)", "Период лактации"]
berem_time_types=["первые 4 недедели беременности","последние 5 недель беременности"]
lact_time_types=["1 неделя","2 неделя","3 неделя","4 неделя"]
age_category_types=["puppy","adult","senior"]
size_types=["small",  "medium",  "large"]

def show_dog_characterictics(disease_df):
    st.set_page_config(page_title="Рекомендации по питанию собак", layout="centered")
    st.header("Рекомендации по питанию собак")
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
              
    show_sidebar()
    breed_list = sorted(disease_df["name_breed"].unique())
    user_breed = st.selectbox("Порода собаки:", breed_list)
    breed_size, avg_wight = size_category(disease_df[disease_df["name_breed"] == user_breed])
    age_type_categ = age_type_category(breed_size, age ,age_metric)
      
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
          
    return user_breed, breed_size, avg_wight, age_type_categ
