# Restaurant Recommendation System

## Cognifyz Machine Learning Internship - Task 2

### Objective
Build a restaurant recommendation system based on user preferences such as cuisine and price range.

### Dataset
The dataset contains 9,551 restaurant records with information such as:

- Restaurant Name
- City
- Cuisines
- Price Range
- Average Cost for Two
- Aggregate Rating
- Votes

### Preprocessing
The following preprocessing steps were performed:

- Handled missing cuisine values.
- Replaced missing cuisine entries with `Unknown`.
- Removed duplicate restaurant records.
- Reset the dataset index.

### Recommendation Approach
A content-based filtering approach was used.

The recommendation system works using:

- TF-IDF Vectorization for cuisine features
- Cosine Similarity to compare user cuisine preferences with restaurant cuisines
- Price Range filtering
- Aggregate Rating and Votes for ranking recommendations

### Test Cases

#### Test Case 1
Cuisine: Japanese  
Price Range: 3

The system successfully recommended Japanese restaurants matching the selected price range.

#### Test Case 2
Cuisine: North Indian  
Price Range: 2

The system successfully recommended North Indian restaurants matching the selected price range.

### Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

### Result
The restaurant recommendation system successfully generates relevant restaurant recommendations based on user cuisine preference and price range.

### Conclusion
The project demonstrates a simple and effective content-based restaurant recommendation system using machine learning techniques.
