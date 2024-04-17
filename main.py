import json
import numpy as np
import matplotlib.pyplot as plt

# Function to load data from JSON files
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Function to find the most popular words in a given year
def find_most_popular(data, year, top_n):
    popularity_in_year = [(word, data[word][year - 1950]) for word in data]
    popularity_in_year.sort(key=lambda x: x[1], reverse=True)
    return popularity_in_year[:top_n]

# Function to calculate growth of words from 1950 to 2000
def calculate_growth(data, year_start, year_end, top_n):
    growth = {}
    for word in data:
        p_start = data[word][year_start - 1950]
        p_end = data[word][year_end - 1950]
        growth[word] = (p_end - p_start) / p_end

    sorted_growth = sorted(growth.items(), key=lambda x: x[1], reverse=True)
    top_growth = sorted_growth[:top_n]
    bottom_growth = sorted_growth[-top_n:]
    negative_growth_count = sum(1 for g in growth.values() if g < 0)

    return top_growth, bottom_growth, negative_growth_count

# Function to identify verbs peaking in a particular year
def find_peaking_verbs(data, year):
    peaking_verbs = [word for word in data if data[word][year - 1951] < data[word][year - 1950] > data[word][year - 1949]]
    return peaking_verbs

# Function to count peaks for each verb
def count_peaks(data):
    peak_counts = {word: sum(1 for i in range(1, 49) if data[word][i-1] < data[word][i] > data[word][i+1] and data[word][i] > max(data[word][:i]) and data[word][i] > max(data[word][i+1:])) for word in data}
    return peak_counts

# Function to plot histogram of peak counts
def plot_peak_histogram(peak_counts):
    peak_values = list(peak_counts.values())
    plt.hist(peak_values, bins=range(max(peak_values) + 2), align='left', rwidth=0.8)
    plt.xlabel('Number of Peaks')
    plt.ylabel('Number of Verbs')
    plt.title('Histogram of Verb Peak Counts')
    plt.grid(True)
    plt.show()

# Function to calculate rank of words in a given year
def calculate_rank(data, year):
    sorted_data = sorted(data.items(), key=lambda x: x[1][year - 1950], reverse=True)
    ranks = {word: i + 1 for i, (word, _) in enumerate(sorted_data)}
    return ranks

# Function to find most popular words in G(y) and L(y)
def find_words_in_G_and_L(data, ranks, year):
    words_in_G = [word for word in data if ranks[word][year - 1950] < ranks[word][year - 1949]]
    words_in_L = [word for word in data if ranks[word][year - 1950] > ranks[word][year - 1949]]
    return words_in_G[:10], words_in_L[:10]

# Function to plot pie chart for verb percentages in L, G, and neither
def plot_verb_pie_chart(data, ranks, year):
    words_in_G, words_in_L = find_words_in_G_and_L(data, ranks, year)
    neither_count = len(data) - len(words_in_G) - len(words_in_L)
    labels = ['In G', 'In L', 'Neither']
    sizes = [len(words_in_G), len(words_in_L), neither_count]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title(f'Verb Distribution in {year}')
    plt.show()

# Function to plot percentage of verbs in L(y) vs. y
def plot_percentage_of_verbs_in_L(data, ranks):
    years = range(1950, 2000)
    percentages = [(sum(1 for word in data if ranks[word][year - 1950] > ranks[word][year - 1949]) / len(data)) * 100 for year in years]
    plt.plot(years, percentages)
    plt.xlabel('Year')
    plt.ylabel('Percentage of Verbs in L(y)')
    plt.title('Percentage of Verbs in L(y) vs. Year')
    plt.grid(True)
    plt.show()

# Function to plot popularity against rank for top N verbs
def plot_popularity_vs_rank(data, year, top_n):
    ranks = calculate_rank(data, year)
    sorted_data = sorted(data.items(), key=lambda x: x[1][year - 1950], reverse=True)[:top_n]
    words = [word for word, _ in sorted_data]
    popularity = [data[word][year - 1950] for word in words]
    ranks = [ranks[word][year - 1950] for word in words]

    plt.scatter(ranks, popularity)
    plt.xlabel('Rank')
    plt.ylabel('Popularity')
    plt.title(f'Popularity vs. Rank of Top {top_n} Verbs in {year}')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

# Function to fit a line to the popularity vs. rank graph and calculate mean absolute error
def fit_line_to_popularity_vs_rank(data, year, top_n):
    ranks = calculate_rank(data, year)
    sorted_data = sorted(data.items(), key=lambda x: x[1][year - 1950], reverse=True)[:top_n]
    words = [word for word, _ in sorted_data]
    popularity = [data[word][year - 1950] for word in words]
    ranks = [ranks[word][year - 1950] for word in words]

    x = np.array(ranks)
    y = np.array(popularity)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, np.log(y), rcond=None)[0]

    predicted_y = np.exp(m * x + c)
    mean_absolute_error = np.mean(np.abs(y - predicted_y))

    return m, c, mean_absolute_error

# Function to plot popularity against rank for all years and classes
def plot_all_popularity_vs_rank(data, top_n):
    years = range(1950, 2000)
    for year in years:
        for word_class in ['verbs', 'nouns', 'adjectives', 'adverbs']:
            class_data = load_data(f'{word_class}.json')
            plot_popularity_vs_rank(class_data, year, top_n)

# Load data
nouns_data = load_data('nouns.json')
verbs_data = load_data('verbs.json')
adjectives_data = load_data('adjectives.json')
adverbs_data = load_data('adverbs.json')

# Question 1
print("Question 1:")
print("Ten most popular nouns of 1990:")
print(find_most_popular(nouns_data, 1990, 10))

print("Popular verbs of 1950:")
print(find_most_popular(verbs_data, 1950, 10))

print("Popular verbs of 2000:")
print(find_most_popular(verbs_data, 2000, 10))

print("Popular adverbs of 1991:")
print(find_most_popular(adverbs_data, 1991, 10))

print("Popular adjectives of 1968:")
print(find_most_popular(adjectives_data, 1968, 10))

# Question 2
print("\nQuestion 2:")
print("Growth of verbs from 1950 to 2000:")
top_growth_verbs, bottom_growth_verbs, negative_growth_count_verbs = calculate_growth(verbs_data, 1950, 2000, 1000)
print("Top growth verbs:")
print(top_growth_verbs[:10])
print("Bottom growth verbs:")
print(bottom_growth_verbs[:10])
print("Number of verbs with negative growth:", negative_growth_count_verbs)

print("Growth of adjectives from 1950 to 2000:")
top_growth_adjectives, bottom_growth_adjectives, negative_growth_count_adjectives = calculate_growth(adjectives_data, 1950, 2000, 1000)
print("Top growth adjectives:")
print(top_growth_adjectives[:10])
print("Bottom growth adjectives:")
print(bottom_growth_adjectives[:10])
print("Number of adjectives with negative growth:", negative_growth_count_adjectives)

print("Growth of adverbs from 1950 to 2000:")
top_growth_adverbs, bottom_growth_adverbs, negative_growth_count_adverbs = calculate_growth(adverbs_data, 1950, 2000, 1000)
print("Top growth adverbs:")
print(top_growth_adverbs[:10])
print("Bottom growth adverbs:")
print(bottom_growth_adverbs[:10])
print("Number of adverbs with negative growth:", negative_growth_count_adverbs)

# Identify the class with the most negative-growth words
negative_growth_counts = {
    'Verbs': negative_growth_count_verbs,
    'Adjectives': negative_growth_count_adjectives,
    'Adverbs': negative_growth_count_adverbs
}
class_with_most_negative_growth = max(negative_growth_counts, key=negative_growth_counts.get)
print("Class with the most negative-growth words:", class_with_most_negative_growth)

# Question 3
print("\nQuestion 3:")
print("Computing |SP(y)| at y = 1950 and y = 2000:")
peak_counts_1950 = count_peaks(verbs_data)
peak_counts_2000 = count_peaks(verbs_data)
print("Number of verbs peaking in 1950:", len([word for word in peak_counts_1950 if peak_counts_1950[word] > 0]))
print("Number of verbs peaking in 2000:", len([word for word in peak_counts_2000 if peak_counts_2000[word] > 0]))

print("Plotting |S(y)| against y:")
plot_peak_histogram(peak_counts_1950)
plot_peak_histogram(peak_counts_2000)

print("Printing popular verbs with a peak in 1991:")
peaking_verbs_1991 = find_peaking_verbs(verbs_data, 1991)
print(find_most_popular(verbs_data, 1991, 10))

print("Percentage of verbs with no peaks at all:", len([word for word in peak_counts_1950 if peak_counts_1950[word] == 0]) / len(peak_counts_1950) * 100)

print("Plotting a histogram of the number of peaks:")
plot_peak_histogram(peak_counts_1950)

# Question 4
print("\nQuestion 4:")
ranks_1981 = calculate_rank(verbs_data, 1981)
print("Ten most popular verbs in G(1981):")
print(find_words_in_G_and_L(verbs_data, ranks_1981, 1981)[0])

print("Ten most popular verbs in L(1981):")
print(find_words_in_G_and_L(verbs_data, ranks_1981, 1981)[1])

plot_verb_pie_chart(verbs_data, ranks_1981, 1960)
plot_percentage_of_verbs_in_L(verbs_data, ranks_1981)
plot_all_popularity_vs_rank(verbs_data, 100)

# Question 5
print("\nQuestion 5:")
plot_popularity_vs_rank(verbs_data, 2000, 100)
plot_popularity_vs_rank(verbs_data, 2000, 1000)
m, c, mean_absolute_error = fit_line_to_popularity_vs_rank(verbs_data, 2000, 100)
print("Values of a and b for the line of best fit:", m, c)
print("Mean absolute error of the regression line:", mean_absolute_error)

plot_all_popularity_vs_rank(verbs_data, 100)

