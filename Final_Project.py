import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree

def load_csv(file):
    """
    return a Data Frame with the data from the given 'file' csv file
    :param file: csv file
    :return: Data Frame
    """
    df = pd.read_csv(file)
    return df


def remove_duplicatives(df, col_name=None):
    """
    return a copy of the given 'df' dataframe, after removing the duplicatives.
    :param df: Data Frame
    :param col_name: If you want to delete duplicates from specific column
    :return: Data frame
    """
    if col_name != None:
        return df.drop_duplicates(subset=col_name)
    else:
        return df.drop_duplicates()


def remove_corrupt_rows(df, num_max_missing_cols):
    """
    return a copy of the given 'df' dataframe, after removing rows with more than 'num_max_missing_cols' missing columns
    :param df: Data Frame
    :param num_max_missing_cols: integer
    :return: Data frame
    """
    relevant_indices = df.isna().sum(axis=1) <= num_max_missing_cols

    df_copy = df.iloc[np.array(relevant_indices), :]

    return df_copy


def replace_missing_values(df, col_to_def_val_dict):
    """
    return a copy of the given 'df' dataframe, after replacing NaN values.
    For numeric data will replace with the median value of the column.
    For Non numeric data will replace with the given default value in col_to_def_val_dict dictionary.
    :param df: Data Frame
    :param col_to_def_val_dict: Dictionary
    :return: Data frame
    """
    df_copy = df.copy()
    dict_cols = []
    for col in col_to_def_val_dict:
        dict_cols.append(col)
        df_copy[col].fillna(col_to_def_val_dict[col], inplace=True)
    num_cols = df_copy._get_numeric_data().columns
    for col in num_cols:
        df_copy[col].fillna(df_copy[col].median(), inplace=True)
    cat_cols = list(set(df.columns) - set(num_cols) - set(dict_cols))
    for col in cat_cols:
        df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
    return df_copy


def outlier_detection_iqr(df):
    """
    return a copy of the given dataframe, after detecting outlier values using IQR
    :param df: Data Frame
    :return: Data frame
    """
    new_df = df.copy()

    numeric_columns = ['year_film', 'year_ceremony', 'ceremony']

    for col in numeric_columns[:-1]:
        Q1 = np.percentile(new_df[col], 25)
        Q3 = np.percentile(new_df[col], 75)
        IQR = Q3 - Q1
        lower_range = Q1 - 1.5 * IQR
        upper_range = Q3 + 1.5 * IQR

        new_df.loc[((new_df[col] > upper_range) | (new_df[col] < lower_range)), col] = np.NAN

    return new_df

if __name__ == '__main__':
# loading the csv file into a data frame
    file = "the_oscar_award.csv"
    df = load_csv(file)

# cleaning the data using the data methods above
    df = remove_duplicatives(df)
    num_max_missing_cols = 2
    df = remove_corrupt_rows(df, num_max_missing_cols)
    col_to_def_val = {'category': 'unknown', 'name': 'unknown', 'film': 'unknown', 'winner': 'FALSE'}
    df = replace_missing_values(df, col_to_def_val)

# Exploratory data analysis

# bar plot
    data = dict(df["winner"].value_counts())
    did_win = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(10, 5))
# creating the bar plot
    plt.bar(did_win, values, color='lightblue', width=0.4)

    plt.xlabel("Has won")
    plt.ylabel("No. of wins")
    plt.title("Distribution number of wins along the years")
    plt.show()


# multiple bar plots
# set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))
# set height of bar
    won = [4, 3, 3, 3, 3]
    nomination = [12, 12, 21, 7, 4]
# Set position of bar on X axis
    br1 = np.arange(len(won))
    br2 = [x + barWidth for x in br1]
# Make the plot
    plt.bar(br1, won, color='g', width=barWidth,
            edgecolor='grey', label='won')
    plt.bar(br2, nomination, color='r', width=barWidth,
            edgecolor='grey', label='nomination')

    plt.title("Number of Wins Vs. Nominations")
    plt.xticks([r + barWidth for r in range(len(won))],
               ['Katharine Hepburn', 'Jack Nicholson', 'Meryl Streep', 'Ingrid Bergman', 'Walter Brennan'])
    plt.legend()
    plt.show()


# pie plot
    sr_category = pd.Series([4, 4, 4, 3, 3, 3, 2, 2, 2, 2],index=['DIRECTING', 'SOUND EDITING', 'SOUND MIXING', 'MUSIC (Original Score)',
                                                                  'FILM EDITING', 'CINEMATOGRAPHY', 'VISUAL EFFECTS',
                                                                  'COSTUME DESIGN', 'PRODUCTION DESIGN', 'BEST PICTURE'],
                            name="category")
    sr_category.plot(kind="pie", rot=0)
    plt.show()


# Machine learning
# encoding the data to have only numbers
    label_encoder = LabelEncoder()
    df['category'] = label_encoder.fit_transform(df['category'])
    df['name'] = label_encoder.fit_transform(df['name'])
    df['film'] = label_encoder.fit_transform(df['film'])

# replacing true to 1 value and false to 0 value
    df.iloc[df['winner'] == False, -1] = 0
    df.iloc[df['winner'] == True, -1] = 1

# logistic regression
# define x and y for logistic regression
    X = df.loc[:, ["year_film", "category", "name", "film"]]
    y = df.loc[:, "winner"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# building the model for logistic regression
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_predictions = clf.predict(X_test)

# Printing the confusion matrix and prediction rate for logistic regression
    print("confusion matrix for logistic regression is:\n", confusion_matrix(y_test, y_predictions))
    print("accuracy_score of your model is:", round(accuracy_score(y_test, y_predictions) * 100, 2), "%")


# decision tree
# define x and y for decision tree
    X2 = df.loc[:, ["year_film", "name", "film", "winner"]]
    y2 = df.loc[:, "category"]
    X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size=0.2, random_state=42)

# building the model for decision tree
    clf2 = tree.DecisionTreeClassifier()
    clf2 = clf2.fit(X2_train, y2_train)
    y_cat_predictions = clf2.predict(X2_test)

# Printing the confusion matrix and prediction rate for decision tree
    print("confusion matrix for decision tree is:\n", confusion_matrix(y2_test, y_cat_predictions))
    print("accuracy_score of your model is: ", round(accuracy_score(y2_test, y_cat_predictions) * 100, 2), "%")

