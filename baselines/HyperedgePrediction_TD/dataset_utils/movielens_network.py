from dataset_utils.aminer_network import prune_hyedges


def get_movielens_network(dataset_folder):
    movie_ids = set([])

    hyedges = []

    director_movieids = {}

    movieid_genreid = {}

    genrename_genreid = {}

    movies_file_handle = open(dataset_folder + "/movies.dat", encoding="ISO-8859-1")
    next(movies_file_handle)
    for line in movies_file_handle.readlines():
        temp = line.split("\n")
        line_split = temp[0].split("\t")

        movie_ids.add(line_split[0])

    directors_file_handle = open(dataset_folder + "/movie_directors.dat", encoding="ISO-8859-1")
    next(directors_file_handle)
    for line in directors_file_handle.readlines():
        temp = line.split("\n")
        line_split = temp[0].split("\t")

        movieid = line_split[0]
        directorid = line_split[1]

        if directorid not in director_movieids:
            director_movieids[directorid] = []

        director_movieids[directorid].append(movieid)

    genre_file_handle = open(dataset_folder + "/movie_genres.dat", encoding="ISO-8859-1")
    next(genre_file_handle)
    for line in genre_file_handle.readlines():
        temp = line.split("\n")
        line_split = temp[0].split("\t")

        movieid = line_split[0]
        genre = line_split[1]

        if genre not in genrename_genreid:
            genreid = len(genrename_genreid)
            genrename_genreid[genre] = genreid

        genreid = genrename_genreid[genre]

        if movieid not in movieid_genreid:
            movieid_genreid[movieid] = []

        movieid_genreid[movieid].append(genreid)

    for directorid in director_movieids:
        hyedges.append(list(director_movieids[directorid]))
        
    pruned_hyedges = prune_hyedges(movieid_genreid.keys(), movie_ids, hyedges)

    return pruned_hyedges
