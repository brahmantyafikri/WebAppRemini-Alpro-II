import ssl
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse

app = Flask(__name__)

# Load your dataset
recipes = pd.read_csv('NewFile.csv')



def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())

def mcdm(recipes, rating_weight, total_time_weight):
    # Sesuaikan nilai agar semakin rendah semakin baik
    recipes['norm_total_time'] = recipes['total_time'].max() - recipes['total_time']

    # Sesuaikan nilai agar semakin tinggi semakin baik
    recipes['mcdm_score'] = (
        rating_weight * normalize_column(recipes['rating']) +
        total_time_weight * recipes['norm_total_time']
    )
    return recipes

def input_dropdown(prompt, options, default=''):
    while True:
        print(prompt)
        for i, option in enumerate(options, start=1):
            print(f"{i}. {option}")

        choice = input("Pilih nomor opsi (tekan Enter untuk default): ")
        if not choice:
            return default
        elif choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        else:
            print("Pilihan tidak valid. Silakan pilih nomor opsi yang benar.")

def print_recipe_info(index, row):
    print(f"\nRekomendasi #{index + 1}:")
    print(f"Recipe Title: {row['recipe_title']}")
    print(f"Ingredients: {row['ingredients']}")
    print(f"Instructions: {row['instructions']}")
    print(f"Total Time: {row['total_time']} minutes")
    print(f"Rating: {row['rating']}")
    # Uncomment the line below if 'similarity_score' is available in your DataFrame
    # print(f"Similarity Score: {row['similarity_score']}")
    print("------------------------------")

def recommend_recipe(recipes, owned_ingredients_input, diet_input, course_input, time_range_input, rating_weight=0.5, total_time_weight=0.5):
    print("Owned Ingredients:", owned_ingredients_input)

    cook_time_min, cook_time_max = time_range_input

    # Ubah bahan yang dimiliki ke huruf kecil untuk pencarian case-insensitive
    owned_ingredients = [ingredient.lower() for ingredient in owned_ingredients_input]

    # Menggabungkan tiap kata pada bahan-bahan menjadi satu string dengan koma sebagai pemisah
    recipes['ingredients_str'] = recipes['ingredients'].apply(lambda x: ', '.join(x.split()))

    # Membuat TfidfVectorizer tanpa stop words
    tfidf_vectorizer = TfidfVectorizer()

    # Menghitung TF-IDF matrix dari data resep dan data bahan yang dimiliki
    tfidf_matrix_recipes = tfidf_vectorizer.fit_transform(recipes['ingredients_str'])

    # Periksa apakah TF-IDF matrix tidak kosong
    if not tfidf_matrix_recipes.nnz:
        return pd.DataFrame()

    # Menghitung TF-IDF matrix dari bahan yang dimiliki dengan referensi TF-IDF matrix resep
    tfidf_matrix_owned = tfidf_vectorizer.transform([' '.join(owned_ingredients)])

    # Pastikan jumlah fitur pada TF-IDF matrix bahan yang dimiliki sesuai dengan resep
    if tfidf_matrix_owned.shape[1] < tfidf_matrix_recipes.shape[1]:
        tfidf_matrix_owned = scipy.sparse.hstack([
            tfidf_matrix_owned,
            scipy.sparse.csr_matrix((tfidf_matrix_owned.shape[0], tfidf_matrix_recipes.shape[1] - tfidf_matrix_owned.shape[1]))
        ])

    # Menghitung similarity score antara data resep dan data bahan yang dimiliki
    cosine_sim = cosine_similarity(tfidf_matrix_owned, tfidf_matrix_recipes)

    # Menambahkan kolom similarity_score ke DataFrame
    recipes['similarity_score'] = cosine_sim[0]

    # Filter resep berdasarkan similarity score
    filtered_data = recipes[recipes['similarity_score'] > 0]

    # Sesuaikan nilai minimum rating jika perlu
    min_rating = 3.5
    filtered_data = filtered_data[filtered_data['rating'] >= min_rating]

    # Filter berdasarkan diet, course, dan waktu masak
    filtered_data = filtered_data[
        (filtered_data['diet'] == diet_input) &
        (filtered_data['course'] == course_input) &
        (filtered_data['total_time'] >= cook_time_min) & (filtered_data['total_time'] <= cook_time_max)
    ]

    if filtered_data.empty:
        # Jika tidak ada yang sepenuhnya cocok, ambil yang paling mendekati berdasarkan similarity score
        closest_data = recipes.loc[recipes['similarity_score'] > 0]
        closest_data = closest_data.sort_values(by='similarity_score', ascending=False).head(5)

        if closest_data.empty:
            return pd.DataFrame()

        # Lanjutkan dengan kode MCDM untuk resep yang paling mendekati
        sorted_recipe = mcdm(closest_data, rating_weight=rating_weight, total_time_weight=total_time_weight)

        # Urutkan berdasarkan skor MCDM
        sorted_recipe = sorted_recipe.sort_values(by='mcdm_score', ascending=False)

        # Print formatted results
        for index, row in sorted_recipe.iterrows():
            print_recipe_info(index, row)

        return sorted_recipe.head(5)

    # Urutkan berdasarkan similarity score
    filtered_data = filtered_data.sort_values(by='similarity_score', ascending=False)

    # Pilih top 6 resep
    filtered_data = filtered_data.head(6)

    # Lanjutkan dengan kode MCDM...
    sorted_recipe = mcdm(filtered_data, rating_weight=rating_weight, total_time_weight=total_time_weight)

    # Urutkan berdasarkan skor MCDM
    sorted_recipe = sorted_recipe.sort_values(by='mcdm_score', ascending=False)

    # Print formatted results
    for index, row in sorted_recipe.iterrows():
        print_recipe_info(index, row)

    return sorted_recipe.head(6)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Define route for the form
@app.route('/resep')
def recipe():
   return render_template("recipe.html")

@app.route('/view_recipe', methods=['POST'])
def view_recipe():
    try:
        # Fetch recipe index from the form
        recipe_index = int(request.form.get('recipe_index'))

        # Fetch the selected recipe details
        selected_recipe = recipes.loc[recipe_index]

        # Render the recipe details template
        return render_template('recipe_details.html', recipe=selected_recipe)

    except Exception as e:
        # Handle exceptions (e.g., invalid input)
        return render_template('error.html', error=str(e))


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Fetch user input from the form
        owned_ingredients_input = request.form.get('owned_ingredients').split(', ')
        diet_input = request.form.get('diet')
        course_input = request.form.get('course')
        time_range_input = tuple(map(int, request.form.get('time_range').split('-')))

        # Call the recommendation function
        recommendations = recommend_recipe(recipes, owned_ingredients_input, diet_input, course_input, time_range_input)

        # Render the results template with the recommendations
        return render_template('results.html', recommendations=recommendations)

    except Exception as e:
        # Handle exceptions (e.g., invalid input)
        return render_template('error.html', error=str(e))



if __name__ == '__main__':
    app.run(debug=True)
