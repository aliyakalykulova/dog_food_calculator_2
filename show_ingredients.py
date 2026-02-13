import streamlit as st
import numpy as np
import pandas as pd
import itertools
from scipy.sparse import hstack, csr_matrix
from kcal_calculate import  apply_category_masks

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


def ingredient_recomendation(ingredient_models,breed_size, age_type_categ,disorder_type, selected_disorder,vectorizer,svd,encoder, df_standart):
            keywords = disorder_keywords.get(disorder_type, selected_disorder).lower()
            kw_tfidf = vectorizer.transform([keywords])
            kw_reduced = svd.transform(kw_tfidf)

            # One-hot for (breed_size, age_type_categ)
            cat_vec = encoder.transform([[breed_size, age_type_categ]])
            kw_combined = hstack([csr_matrix(kw_reduced), cat_vec])
			
            ing_scores = {ing: clf.decision_function(kw_combined)[0] for ing, clf in ingredient_models.items()}

            proteins=df_standart[df_standart["category_ru"].isin(["Мясо","Яйца и Молочные продукты"])]["name_feed_ingredient"].tolist()
            oils=df_standart[df_standart["category_ru"].isin([ "Масло и жир"])]["name_feed_ingredient"].tolist()
            carbonates_cer=df_standart[df_standart["category_ru"].isin(["Крупы"])]["name_feed_ingredient"].tolist()
            carbonates_veg=df_standart[df_standart["category_ru"].isin(["Зелень и специи","Овощи и фрукты"])]["name_feed_ingredient"].tolist()
            water=["water"]
  
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
            st.subheader("🌿 Рекомендуемые ингредиенты")
            for ing in ingredients_finish:
                st.write("• " + ing)
            return ingredients_finish,keywords

def nutrients_recomendation(vectorizer_wet,keywords,svd_wet,encoder_wet, breed_size, age_type_categ, ridge_models,scalers ):
            kw_tfidf = vectorizer_wet.transform([keywords])
            kw_reduced = svd_wet.transform(kw_tfidf)
            cat_vec = encoder_wet.transform([[breed_size, age_type_categ]])
            cat_vec = apply_category_masks(cat_vec,encoder_wet)
            kw_combined = hstack([csr_matrix(kw_reduced), cat_vec])
            nutrient_preds = {}
            for nut, model in ridge_models.items():
                      pred = model.predict(kw_combined)[0]
                      sc = scalers.get(nut)
                      if sc:
                        pred = sc.inverse_transform([[pred]])[0][0]
                      nutrient_preds[nut] = float(round(pred, 2))
            return nutrient_preds




