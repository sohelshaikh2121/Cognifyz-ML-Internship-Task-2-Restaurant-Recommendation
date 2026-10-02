import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# TASK 2 - RESTAURANT RECOMMENDATION SYSTEM
# =========================================================

print("=" * 70)
print("TASK 2 - RESTAURANT RECOMMENDATION SYSTEM")
print("=" * 70)


# =========================================================
# STEP 1 - LOAD DATASET
# =========================================================

df = pd.read_csv("Dataset .csv")

print("\nDataset Shape:")
print(df.shape)


# =========================================================
# STEP 2 - DISPLAY IMPORTANT COLUMNS
# =========================================================

print("\nSelected Columns:")

print(
    df[
        [
            "Restaurant Name",
            "City",
            "Cuisines",
            "Price range",
            "Average Cost for two",
            "Aggregate rating",
            "Votes"
        ]
    ].head()
)


# =========================================================
# STEP 3 - PREPROCESSING
# =========================================================

print("\nMissing values in Cuisines:")
print(df["Cuisines"].isnull().sum())

# Handle missing values
df["Cuisines"] = df["Cuisines"].fillna("Unknown")

# Remove duplicate restaurants
df = df.drop_duplicates(subset=["Restaurant ID"])

# Reset index
df = df.reset_index(drop=True)

print("\nAfter preprocessing:")
print("Total Restaurants:", len(df))
print("Missing Cuisines:", df["Cuisines"].isnull().sum())

print("\nPREPROCESSING COMPLETED")


# =========================================================
# STEP 4 - TF-IDF FEATURE EXTRACTION
# =========================================================

tfidf = TfidfVectorizer()

cuisine_matrix = tfidf.fit_transform(df["Cuisines"])

print("\nTF-IDF FEATURE EXTRACTION COMPLETED")

print("Cuisine Feature Matrix Shape:", cuisine_matrix.shape)


# =========================================================
# STEP 5 - RESTAURANT RECOMMENDATION FUNCTION
# =========================================================

def recommend_restaurants(
    preferred_cuisine,
    preferred_price_range,
    top_n=5
):

    # Filter restaurants according to price range
    filtered_df = df[
        df["Price range"] == preferred_price_range
    ].copy()

    if filtered_df.empty:
        print("\nNo restaurants found for this price range.")
        return

    # Convert user's preferred cuisine into TF-IDF vector
    user_vector = tfidf.transform([preferred_cuisine])

    # Get cuisine vectors of filtered restaurants
    filtered_matrix = cuisine_matrix[filtered_df.index]

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        user_vector,
        filtered_matrix
    ).flatten()

    # Store similarity scores
    filtered_df["Similarity Score"] = similarity_scores

    # Keep restaurants that have cuisine similarity
    matched_df = filtered_df[
        filtered_df["Similarity Score"] > 0
    ].copy()

    if matched_df.empty:
        print(
            "\nNo matching restaurants found for cuisine:",
            preferred_cuisine
        )
        return

    # Sort according to similarity and rating
    recommendations = matched_df.sort_values(
        by=[
            "Similarity Score",
            "Aggregate rating",
            "Votes"
        ],
        ascending=[
            False,
            False,
            False
        ]
    ).head(top_n)

    print("\n" + "=" * 70)
    print("USER PREFERENCE")
    print("=" * 70)

    print("Cuisine:", preferred_cuisine)
    print("Price Range:", preferred_price_range)

    print("\n" + "=" * 70)
    print("TOP RESTAURANT RECOMMENDATIONS")
    print("=" * 70)

    print(
        recommendations[
            [
                "Restaurant Name",
                "City",
                "Cuisines",
                "Price range",
                "Aggregate rating",
                "Votes",
                "Similarity Score"
            ]
        ].to_string(index=False)
    )

    print("\nRecommendation generated successfully.")


# =========================================================
# STEP 6 - TEST CASE 1
# =========================================================

print("\n\nTEST CASE 1")

recommend_restaurants(
    preferred_cuisine="Japanese",
    preferred_price_range=3,
    top_n=5
)


# =========================================================
# STEP 7 - TEST CASE 2
# =========================================================

print("\n\nTEST CASE 2")

recommend_restaurants(
    preferred_cuisine="North Indian",
    preferred_price_range=2,
    top_n=5
)


# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n" + "=" * 70)
print("TASK 2 COMPLETED SUCCESSFULLY")
print("=" * 70)