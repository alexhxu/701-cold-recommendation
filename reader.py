import numpy as np
import pandas as pd
movies_path = "movies.dat"
ratings_path = "ratings.dat"
supp_path = "imdb_supp.csv"

imdb_data = pd.read_csv(supp_path)
imdb_data['movie_code'] = imdb_data['full_name_fstring'] = imdb_data.apply(lambda row: f"{row['original_title']} ({row['year']})", axis=1)


movie_code_list = list()
def create_films_dict(file_path):
    master_dict = {}
    with open(file_path, "r", encoding="latin-1") as file:
        for line in file:
            full_line = line.strip()
            id = int(full_line.split("::")[0])
            movie_code = full_line.split("::")[1]
            title = movie_code[0:-7]
            year = movie_code.split(" (")[-1][0:-1]
            char_to_find = ","
            if char_to_find in title:
                last_word = title.split(",")[-1].strip()
                if last_word == "The" or last_word == "A" or last_word == "An":
                    if len(title.split(",")) > 2:
                        true_title = f"{last_word} {title[0:-5]}"
                    else:
                        last_part = title.split(",")[0]
                        true_title = f"{last_word} {last_part}" 
                    movie_code = f"{true_title} ({year})"
            imdb_filtered = imdb_data[imdb_data['movie_code'] == movie_code]
            info_stump = f"movieID:{id}; title:{movie_code}; year:{year}"
            if(imdb_filtered.shape[0] > 0):
                dur = imdb_filtered['duration'].iloc[0]
                coun = imdb_filtered['country'].iloc[0]
                lan = imdb_filtered['language'].iloc[0]
                dir = imdb_filtered['director'].iloc[0]
                wri = imdb_filtered['writer'].iloc[0]
                prod = imdb_filtered['production_company'].iloc[0]
                act = imdb_filtered['actors'].iloc[0]
                desc = imdb_filtered['description'].iloc[0]
                info_stump = f"{info_stump}; duration:{dur}; country{coun}; language:{lan}; director:{dir}; writers:{wri}; production_company:{prod}; actors:{act}; description:{desc}"
            master_dict[id] = info_stump
    return master_dict

def create_ratings_matrix(file_path):
    full_array = []
    with open(file_path, "r", encoding = "latin-1") as file:
        for line in file:
            full_line = line.strip()
            full_array.append(full_line)
        as_np_array = np.array(full_array)
    return as_np_array
filmsdict = create_films_dict(movies_path)
ratingsmat = create_ratings_matrix(ratings_path)


      