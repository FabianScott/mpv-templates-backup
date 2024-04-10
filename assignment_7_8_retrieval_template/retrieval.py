import numpy as np, time, datetime, scipy.io, os, pdb, pickle
from scipy.sparse import csr_matrix
import PIL.Image
from utils import get_pts_in_box, draw_bbox, vis_results, get_A_matrix_from_geom, get_query_data, get_shortlist_data


def create_db(image_visual_words, num_visual_words, idf):
    """
    create the image database with an inverted file represented as a sparse matrix. 
    the sparse matrix has dimension number_of_visual_words x number_of_images
    the stored representation should be l2 normalized

    image_visual_words: list of arrays indicating the visual words present in each image
    num_visual_words: total number of visual words in the visual codebook
    idf: array with idf weights per visual word
    return -> 
    db: sparse matrix representing the inverted file 
    """
    row_indices = []
    col_indices = []
    data = []

    # Iterate through each image and populate the inverted file
    for i, words in enumerate(image_visual_words):
        # Count occurrence of each visual word in the image
        word_counts = np.bincount(words, minlength=num_visual_words)
        weighted_counts = word_counts * idf
        # Calculate L2 norm
        l2_norm = np.linalg.norm(weighted_counts)
        # Normalize the counts
        normalized_counts = weighted_counts / l2_norm if l2_norm != 0 else weighted_counts
        # Store non-zero entries for the CSR matrix
        non_zero_indices = np.nonzero(normalized_counts)[0]
        row_indices.extend(non_zero_indices)
        col_indices.extend([i] * len(non_zero_indices))
        data.extend(normalized_counts[non_zero_indices])

    # Construct CSR matrix directly
    db = csr_matrix((data, (row_indices, col_indices)), shape=(num_visual_words, len(image_visual_words)),
                    dtype=np.float32)

    return db


def get_idf(image_visual_words, num_visual_words):
    """
    Calculate the IDF weight for visual word

    image_visual_words: list of arrays indicating the visual words present in each image
    num_visual_words: total number of visual words in the visual codebook
    return -> 
    idf: array with idf weights per visual word
    """

    df = np.zeros(num_visual_words)
    idf = np.zeros(num_visual_words)

    for words in image_visual_words:
        df[words] += 1

    idf[df > 0] = np.log(len(image_visual_words) / df[df > 0])

    return idf


def retrieve(db, query_visual_words, idf):
    """
    Search the database with a query, sort images base on similarity to the query. 
    Returns ranked list of image ids and the corresponding similarities

    db: image database
    query_visual_words: array with visual words of the query image
    idf: idf weights
    return -> 
    ranking: sorted list of image ids based on similarities to the query
    sim: sorted list of similarities
    """
    # Create the one-hot vector
    query_vector = np.bincount(query_visual_words.flatten(), minlength=db.shape[0])
    # Weight by idf and normalise
    query_vector = query_vector * idf
    query_vector = query_vector / np.linalg.norm(query_vector)
    sim = query_vector @ db
    ranking = np.argsort(sim)[::-1]

    return ranking, sim[ranking]


def get_tentative_correspondences(query_visual_words, shortlist_visual_words):
    """
    query_visual_words: 1D array with visual words of the query 
    shortlist_visual_words: list of 1D arrays with visual words of top-ranked images 
    return -> 
    correspondences: list of lists of correspondences
    """

    correspondences = []

    for i, q_word in enumerate(query_visual_words):  # loop over the provided list of DB images

        corr = []
        for j, s_word in enumerate(shortlist_visual_words):
            if q_word.item() in s_word:
                corr.append([i, j])

        correspondences.append(corr)

    return correspondences


def ransac_affine(query_geometry, shortlist_geometry, correspondences, inlier_threshold):
    """

    query_geometry: 2D array with geometry of the query
    shortlist_geometry: list of 2D arrays with geometry of top-ranked images
    correspondences: list of lists of correspondences
    inlier_threshold: threshold for inliers of the spatial verification
    return -> 
    inlier_counts: 1D array with number of inliers per image
    transformations: 3D array with the transformation per image
    """

    K = len(shortlist_geometry)
    transformations = np.zeros((K, 3, 3))
    inliers_counts = np.zeros((K,))

    for k in range(K):
        best_score = 0
        A_best = None

        corr = np.array(correspondences[k])
        N = len(corr)

        for n in range(N):
            q_id = corr[n, 0]
            d_id = corr[n, 1]

            Aq = get_A_matrix_from_geom(query_geometry[q_id])  # shape of local feature from the query
            Ad = get_A_matrix_from_geom(shortlist_geometry[k][d_id])  # shape of local feature from DB image

            A = Ad @ np.linalg.inv(Aq)
            transformed_query_points = (A @ query_geometry.reshape((-1, 3)).T)
            distances = np.linalg.norm(transformed_query_points - shortlist_geometry[k], axis=1)
            number_of_inliers = np.sum(distances < inlier_threshold)

            if number_of_inliers > best_score:
                best_score = number_of_inliers
                A_best = A

        inliers_counts[k] = best_score
        transformations[k] = A_best

    return transformations, inliers_counts


def search_spatial_verification(query_visual_words, query_geometry, candidatelist_visual_words,
                                candidatelist_geometries, inlier_threshold):
    """

    query_visual_words: 1D array with visual words of the query 
    query_geometry: 2D array with geometry of the query
    candidatelist_visual_words: list of 1D arrays with visual words of top-ranked images 
    candidatelist_geometry: list of 2D arrays with geometry of top-ranked images
    inlier_threshold: threshold for inliers of the spatial verification

    Returns:
    inlier_counts: 1D array with number of inliers per image
    transformations: 3D array with the transformation per image
    """
    corrs = get_tentative_correspondences(query_visual_words, candidatelist_visual_words)
    transformations, inliers_counts = ransac_affine(query_geometry, candidatelist_geometries, corrs, inlier_threshold)
    return inliers_counts, transformations


### ========================================================
def main():
    include_lab_assignment_2 = True  # set to True for the second part - spatial verif.

    with open('data/mpv_lab_retrieval_data.pkl', 'rb') as handle:
        p = pickle.load(handle)

    visual_words = p['visual_words']
    geometries = p['geometries']
    img_names = p['img_names']
    img_names = ['imgs/' + x + '.jpg' for x in img_names]
    print(len(img_names))
    num_visual_words = 50000 + 1  # for the codebook we used to generate the provided visual words

    # spatial verification parameters
    shortlist_size = 50
    inlier_threshold = 8

    t = time.time()
    idf = get_idf(visual_words, num_visual_words)
    db = create_db(visual_words, num_visual_words, idf)
    print("DB created in {:.5}s".format(time.time() - t))

    q_id = 347  # pick a query
    t = time.time()
    ranked_img_ids, scores = retrieve(db, visual_words[q_id], idf)
    print("query performed in {:.3f}s".format(time.time() - t))
    print(ranked_img_ids[:10], scores[:10])

    if include_lab_assignment_2:
        bbox_xyxy = [350, 200, 700, 600]  # pick a bounding box
        query_visual_words_inbox, query_geometry_inbox = get_query_data(visual_words, geometries, q_id, bbox_xyxy)
        t = time.time()
        ranked_img_ids, scores = retrieve(db, query_visual_words_inbox, idf)
        print("query-cropped performed in {:.3f}s".format(time.time() - t))
        print(ranked_img_ids[:10], scores[:10])

        shortlist_ids = ranked_img_ids[:shortlist_size]  # apply SP only to most similar images
        shortlist_visual_word, shortlist_geometry = get_shortlist_data(visual_words, geometries, shortlist_ids)

        t = time.time()
        scores_sp, transformations = search_spatial_verification(query_visual_words_inbox, query_geometry_inbox,
                                                                 shortlist_visual_word, shortlist_geometry,
                                                                 inlier_threshold)
        print("spatial verification performed in {:.3f}s".format(time.time() - t))

        idxs = np.argsort(-scores_sp)
        scores_sp = scores_sp[idxs]
        transformations = transformations[idxs]
        top_img_ids = ranked_img_ids[idxs]
        print(top_img_ids[:10], scores_sp[:10])

        # will create fig.png - check it out
        vis_results(img_names, q_id, bbox_xyxy, top_img_ids, scores_sp, transformations)


if __name__ == '__main__':
    main()
