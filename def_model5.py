import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

df=pd.read_csv('clean_genre_unique_10_05_noNan.csv')

df.rename(columns={'weighted_rating': 'Rate'}, inplace=True)
df['Rate'] = df['Rate'].apply(lambda x: "{:.2f}".format(x))
df.rename(columns={'genre':'Genre'}, inplace=True)
df['Titre'] = df['Titre'].str.title()
df['Genre'] = df['Genre'].str.title()
df['Directors'] = df['Directors'].str.title()
df.rename(columns={'startYear':'Year'}, inplace=True)

features = ['Genre', 'Titre', 'Rate', 'Directors']



for feature in features:
    df[feature] = df[feature].fillna('')
    
df['index'] = df.index

def get_title_from_index(index):
    return df[df.index == index]["Titre"].values[0]
def get_index_from_title(Titre):
    return df[df.Titre == Titre]["index"].values[0]

def combine_features(row):
    return row['Titre'] + ' ' + row['Genre'] + ' ' + row['Directors'].split(' ', 1)[-1] + ' ' + str(row['Rate']+ ' ' + row['Liste acteurs'])

df['combined_features'] = df.apply(combine_features, axis=1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(count_matrix)


knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(count_matrix)


    



def get_recommendations4(title, genre_input, actor_input, director_input):
    movie_user_likes = title
    movie_index = get_index_from_title(movie_user_likes)
    
    def combine_features(row, genre_input, actor_input, director_input):
        features = [row['Genre'], row['Titre'], row['Directors'].split(' ', 1)[-1], str(row['Rate']), row['Liste acteurs']]
        if genre_input:
            features.extend(genre_input*5)
        if actor_input:
            features.extend([actor.split(' ', 1)[-1] for actor in actor_input])
        if director_input:
            features.extend([director.split(' ', 1)[-1] for director in director_input]*20)
        return ' '.join(features)
    
    # Modifier les features avec les choix de l'utilisateur
    df['combined_features'] = df.apply(combine_features, axis=1, args=(genre_input, actor_input, director_input))
    filtered_count_matrix = cv.transform(df['combined_features'])
    
    similar_movies_cosine = cosine_sim[movie_index]
    
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(filtered_count_matrix)
    _, indices = knn_model.kneighbors(filtered_count_matrix[movie_index], n_neighbors=10)
    
    combined_indices = set(indices.flatten()) | set(similar_movies_cosine.argsort()[::-1][0:11])
    combined_indices.discard(movie_index)
    
    movie_info = []
    
    for index in combined_indices:
        movie_data = df.loc[index, ['Titre', 'Genre', 'Directors', 'Year', 'Rate']]
        movie_info.append(movie_data)
    
    movie_info_df = pd.DataFrame(movie_info, columns=['Titre', 'Genre', 'Directors', 'Year', 'Rate'])
    movie_info_df = movie_info_df.head(5)
    
    return movie_info_df


        
