import streamlit as st

def init_global():
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
			
