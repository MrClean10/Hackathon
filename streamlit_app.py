
import pandas as pd
import streamlit as st
import requests
import random
from time import sleep
from def_model5 import get_recommendations4
from PIL import Image
from citations import citations


df=pd.read_csv('clean_genre_unique_10_05_noNan.csv')
df.rename(columns={'weighted_rating': 'Rate'}, inplace=True)
df['Rate'] = df['Rate'].apply(lambda x: "{:.2f}".format(x))
df.rename(columns={'genre':'Genre'}, inplace=True)
df['Titre'] = df['Titre'].str.title()
df['Genre'] = df['Genre'].str.title()
df['Directors'] = df['Directors'].str.title()
df.rename(columns={'startYear':'Year'}, inplace=True)

titles = df['Titre'].tolist()
titles = sorted(titles)
genres = ['Comedy', 'Drama','Action','Crime','Adventure','Biography','Horror']
genres=sorted(genres)
acteurs = df['Liste acteurs'].str.split(',').tolist()
acteurs_cleaned = [acteur.replace("'", "").replace('"',"").replace("[", "").replace("]", "").strip() for sublist in acteurs for acteur in sublist]
acteurs_unique = list(set(acteurs_cleaned))
acteurs_unique = sorted(acteurs_unique, key=lambda x: (x.split()[-1], x.split()[0]))
acteurs_unique = [f"{' '.join(acteur.split()[::-1])}" for acteur in acteurs_unique]

directors = df['Directors'].unique().tolist()
directors = sorted(directors, key=lambda x: (x.split()[-1], x.split()[0]))
directors = [f"{' '.join(director.split()[::-1])}" for director in directors]

st.set_page_config(layout='wide')

image2=Image.open('Banniere_ticket.png')
st.image(image2,use_column_width=True)

def reset_state():
    st.session_state.pop("title", None)
    st.session_state.pop("genre_input", None)
    st.session_state.pop("actor_input", None)
    st.session_state.pop("director_input", None)
    st.script_request_queue.clear()
    st.experimental_rerun()
   
tab1, tab2 = st.tabs([':cinema: RECOMMANDATIONS :cinema:', ':musical_note::microphone: BLINDTEST :microphone::musical_note:'])

with tab1 :

   col1,col2,col3 = st.columns([1,2,1])
   with col2:
      with st.form("form 1"):
         title = st.selectbox("Saisis un film que tu as aimé **(obligatoire)** : ",options=titles )
         st.markdown("______________________________________________________")
         st.markdown('Pour affiner la recommandation  : ')
         st.markdown(' ')
         genre_input = st.multiselect(label="Sélectionne un ou plusieurs genres  _(optionnel)_ :", options=['No Choice'] + genres)
         actor_input = st.multiselect(label="Sélectionne un ou plusieurs acteurs  _(optionnel)_ :", options=['No Choice'] + acteurs_unique)
         director_input = st.multiselect(label="Sélectionne un ou plusieurs réalisateurs  _(optionnel)_ :", options=['No Choice'] + directors)
         rate = df.loc[df['Titre'] == title, 'Rate'].values[0]
         submitted = st.form_submit_button(":clapper: :popcorn:")


   if submitted:
      col1,col2,col3=st.columns([1,3,1])
      with col1:
             if st.button("Remise à Zéro"):
                 reset_state()
          
      with col2:
          progress_text = random.choice(citations)
          with st.spinner(progress_text):
              sleep(5)
    
      recommendations = get_recommendations4(title, genre_input, actor_input,director_input)
      st.write(' Nous vous recommandons : ' )
      hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
      st.markdown(hide_table_row_index, unsafe_allow_html=True)
      st.table(recommendations)
      
        
      for title2 in recommendations.iloc[:, 0]:
          try:
              url = f"http://www.omdbapi.com/?t={title2}&apikey=daebd117&"
              response = requests.get(url)
              movie_data = response.json()
        
              if str(movie_data['Year']) == str(recommendations.loc[recommendations['Titre'] == title2, 'Year'].values[0]):
                  col1, col2 = st.columns([1, 2])
                  with col1:
                      st.image(movie_data['Poster'])
                  with col2:
                      st.subheader(movie_data['Title'])
                      st.caption(f"Genre: {movie_data['Genre']}  Année: {movie_data['Year']}  Durée: {movie_data['Runtime']}")
                      st.write(movie_data['Plot'])
                      ID = movie_data['imdbID']
                      st.write(f"Plus d'informations sur [IMDb](https://www.imdb.com/title/{ID}/?ref_=fn_al_tt_1)")
              else:
                  st.write('')

          except:
              st.write('')

with tab2 : 
       st.video('https://youtu.be/cYrv98MNNZA')
