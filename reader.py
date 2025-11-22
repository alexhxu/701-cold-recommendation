import numpy as np
movies_path = "movies.dat"
ratings_path = "ratings.dat"

def create_films_dict(file_path):
    master_dict = {}
    with open(file_path, "r", encoding="latin-1") as file:
        for line in file:
            full_line = line.strip()
            id = int(full_line.split("::")[0])
            master_dict[id] = full_line
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


      