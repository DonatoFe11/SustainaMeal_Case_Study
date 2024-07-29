# Introduction
This repo is the used to store the all the data and notebooks of our project called Sustainameal. This is a library that we built which is used to get the most sustainable and healthy alternative recipe to a given one.


To navigate our experiments :
https://giovtemp.github.io/SustainaMeal_Case_Study/experiments/results/experiments.html
https://giovtemp.github.io/SustainaMeal_Case_Study/experiments/results/experiments_unknown_recipes.html
https://giovtemp.github.io/SustainaMeal_Case_Study/experiments/results/experiments_gpt_rerank.html


## notebooks/Preprocessing.ipynb
This file contains the code used to preprocess the dataset of the recipes, introducing also the sustainability score and the sustainability labels, computed through the dataset of ingredients, builded from the CSEL dataset. For the sustainability labels, we split the range of the values in some beans and give them a specific label

## notebooks/EDA_recipes_csv.ipynb
This file contains the code used to explore the dataset of the recipes

## notebooks/Ingredient_dataset_generation.ipynb
This file contains the code used to build the dataset of ingredients, with the values of the carbon foot print and the water foot print, from the CSEL dataset

## notebooks/Labeling_dataset.ipynb
This file contains the code to labels the healthiness score(who_score), so split the range of the values in some beans and give them a specific label

## data/ingredients_dictionary.csv
This file contains the dataset of the ingredients with the relative values of the carbon foot print and the water foot print

## data/final_recipes_set.csv
This file contains the dataset of the recipes

## data/experiment_unkown_recipes_results.json
This file contains the experiment did with recipes not present in our dataset

## data/experiment_results.json
This file contains the experiment did with recipes present in our dataset

## experiments/results/experiments.html
This file is the html page of the results of the experiments with the known recipes

## experiments/results/experiments_unknown_recipes.html
This file si the html page of the results of the experiments with the unknown recipes

## experiments/results/images/risultati_esperimenti_ricette_label_1.png
This image shows the results of the experiments with recipes on average unsustainable

## experiments/results/images/risultati_esperimenti_ricette_label_2.png
This image shows the results of the experiments with recipes highly unsustainable

## experiments/results/images/risultati_esperimenti_ricette_sconosciute.png
This image shows the results of the experiments with the unknown recipes

## experiments/results/images/risultati_esperimenti_ricette_who_score.png
This image shows the results of the experiments with recipes with who_score(healthiness score) above average
