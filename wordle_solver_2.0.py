import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
#from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def download_word_list():
    """Download and prepare the word list."""
    print("Downloading and preparing word list...")
    # Download the file from GitHub
    url = "https://raw.githubusercontent.com/IlyaSemenov/wikipedia-word-frequency/master/results/enwiki-2023-04-13.txt"
    response = requests.get(url)

    # Save the file locally
    with open('enwiki-2023-04-13.txt', 'w', encoding='utf-8') as f:
        f.write(response.text)

    # Read the file into a pandas DataFrame
    df_word_freq = pd.read_csv('enwiki-2023-04-13.txt', sep=' ', names=['word', 'frequency'])

    # Filter for 5-letter words containing only letters
    df_word_freq = df_word_freq[df_word_freq['word'].str.contains('[a-z]', na=False)]
    df_word_freq = df_word_freq[df_word_freq['word'].str.len() == 5]
    df_word_freq = df_word_freq[df_word_freq['word'].str.isalpha() == True]

    # Calculate relative frequency
    max_frequency = df_word_freq['frequency'].max()
    df_word_freq['word_pct'] = ((df_word_freq['frequency']/max_frequency) * 100).round(4)

    print(f"Found {len(df_word_freq)} possible 5-letter words")
    return df_word_freq


def scrape_wordle_words():
    """
    Scrapes past Wordle words from the Rock Paper Shotgun website.
    Returns a pandas DataFrame with only the words.
    """
    # URL of the page containing past Wordle words
    url = "https://www.rockpapershotgun.com/wordle-past-answers"

    # Send HTTP request to the URL
    print("Fetching content from the website...")
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return None

    # Parse HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all elements with class="inline" which contain the Wordle words
    print("Extracting Wordle words...")
    word_elements = soup.find_all(class_="inline")

    # Extract and clean the words
    words = []
    for element in word_elements:
        # Extract text and split by whitespace to get individual words
        text = element.get_text().strip()

        # Filter out 5-letter words (all Wordle words are 5 letters)
        potential_words = [word.strip() for word in re.findall(r'\b\w+\b', text) if len(word.strip()) == 5]

        # Add valid words to our list
        words.extend([word.upper() for word in potential_words])

    # Remove duplicates while preserving order
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)

    # Create DataFrame with only words
    print(f"Creating DataFrame with {len(unique_words)} unique Wordle words...")
    wordle_df = pd.DataFrame({
        'word': unique_words
    })

    return wordle_df


def filter_out_used_words(df_word_freq):
    """
    Filter out words that have already been used in past Wordle puzzles.
    Returns the filtered dataframe.
    """
    print("Removing previously used Wordle words from word list...")

    # Get the list of past Wordle words
    used_words_df = scrape_wordle_words()

    if used_words_df is None or used_words_df.empty:
        print("Warning: Could not retrieve past Wordle words. Using full word list.")
        return df_word_freq

    # Convert all words to lowercase for consistent comparison
    used_words = [word.lower() for word in used_words_df['word']]

    # Filter out the used words
    filtered_df = df_word_freq[~df_word_freq['word'].str.lower().isin(used_words)]

    removed_count = len(df_word_freq) - len(filtered_df)
    print(f"Removed {removed_count} previously used Wordle words.")
    print(f"Remaining available words: {len(filtered_df)}")

    return filtered_df


def filter_words(word_df, known_positions, right_letters, wrong_positions, wrong_letters):
    """Filter words based on wordle feedback."""
    filtered_df = word_df.copy()

    # Filter out words with wrong letters
    if wrong_letters:
        # Exclude wrong letters only if they're not in right_letters
        real_wrong_letters = [letter for letter in wrong_letters if letter not in right_letters]
        if real_wrong_letters:
            pattern = '|'.join(real_wrong_letters)
            filtered_df = filtered_df[~filtered_df['word'].str.contains(pattern)]

    # Filter for words with right letters
    for letter in right_letters:
        filtered_df = filtered_df[filtered_df['word'].str.contains(letter)]

    # Filter for known positions
    for pos, letter in enumerate(known_positions):
        if letter:
            filtered_df = filtered_df[filtered_df['word'].str[pos] == letter]

    # Filter for wrong positions (letter exists but not in this position)
    for pos, letters in enumerate(wrong_positions):
        for letter in letters:
            # Word must contain the letter but not at this position
            filtered_df = filtered_df[
                (filtered_df['word'].str.contains(letter)) &
                (filtered_df['word'].str[pos] != letter)
            ]

    return filtered_df


def calculate_letter_probabilities(word_df):
    """Calculate probabilities for each letter in the remaining words."""
    if len(word_df) == 0:
        return pd.DataFrame(columns=['letter', 'pct_likely'])

    # Create dataframe with alphabet
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    df_letters = pd.DataFrame({'letter': list(alphabet)})

    # Calculate probability for each letter
    for index, row in df_letters.iterrows():
        letter = row['letter']
        words_with_letter = word_df[word_df['word'].str.contains(letter)]
        probability = (len(words_with_letter) / len(word_df)) * 100 if len(word_df) > 0 else 0
        df_letters.at[index, 'pct_likely'] = round(probability, 2)

    return df_letters.sort_values(by=['pct_likely'], ascending=False)


def visualize_top_words(word_df, n=10):
    """Visualize the top N most likely words using Seaborn."""
    if len(word_df) == 0:
        print("No words match the current constraints.")
        return

    top_words = word_df.sort_values(by='word_pct', ascending=False).head(n)

    # Create figure with reduced size (700x280)
    plt.figure(figsize=(7, 2.8))

    # Create the bar plot with Seaborn
    ax = sns.barplot(x=top_words['word'], y=top_words['word_pct'], palette='viridis')

    # Add probability percentage labels on top of each bar
    for i, v in enumerate(top_words['word_pct']):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=9)

    # Set title and labels
    plt.title(f'Top {n} Most Likely Words')
    plt.ylabel('Probability (%)')
    plt.xlabel('')  # Remove x-axis label
    plt.tight_layout()
    plt.show()


def visualize_letter_probabilities(df_letters):
    """Visualize letter probabilities using Seaborn."""
    if len(df_letters) == 0:
        print("No letter probability data available.")
        return

    # Get top 10 letters
    top_letters = df_letters.head(10)

    # Create figure with reduced size (700x280)
    plt.figure(figsize=(7, 2.8))

    # Create the bar plot with Seaborn
    ax = sns.barplot(x=top_letters['letter'], y=top_letters['pct_likely'], palette='coolwarm')

    # Add probability percentage labels on top of each bar
    for i, v in enumerate(top_letters['pct_likely']):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=9)

    # Set title and labels
    plt.title('Most Likely Letters')
    plt.ylabel('Probability (%)')
    plt.xlabel('')  # Remove x-axis label
    plt.tight_layout()
    plt.show()


def display_game_state(known_positions, right_letters, wrong_positions, wrong_letters):
    """Display the current game state."""
    print("\nCurrent Game State:")

    # Display known positions
    display_word = ['_' if not letter else letter for letter in known_positions]
    print(f"Known positions: {' '.join(display_word)}")

    # Display other information
    if right_letters:
        print(f"Correct letters: {', '.join(sorted(set(right_letters)))}")
    if any(wrong_positions):
        all_wrong_pos = []
        for pos, letters in enumerate(wrong_positions):
            if letters:
                all_wrong_pos.extend([f"{letter}≠{pos+1}" for letter in letters])
        print(f"Wrong positions: {', '.join(all_wrong_pos)}")
    if wrong_letters:
        print(f"Wrong letters: {', '.join(sorted(set(wrong_letters)))}")


def process_guess(word, feedback, known_positions, right_letters, wrong_positions, wrong_letters):
    """Process a guess and update game state based on feedback."""
    for i, (letter, result) in enumerate(zip(word, feedback)):
        if result == '2':  # Correct letter, correct position
            known_positions[i] = letter
            if letter not in right_letters:
                right_letters.append(letter)
        elif result == '1':  # Correct letter, wrong position
            if letter not in right_letters:
                right_letters.append(letter)
            wrong_positions[i].append(letter)
        elif result == '0':  # Letter not in word
            if letter not in right_letters:
                wrong_letters.append(letter)


def run_wordle_scraper():
    """Run the wordle scraper standalone functionality."""
    print("Starting Wordle word scraper...")
    wordle_df = scrape_wordle_words()

    if wordle_df is not None and not wordle_df.empty:
        # Display first few words
        print("\nFirst 10 Wordle words:")
        print(wordle_df.head(10))

        # Print some statistics
        print(f"\nTotal unique Wordle words: {len(wordle_df)}")
        print("Done!")
    else:
        print("Failed to create Wordle words DataFrame.")

    return wordle_df


def run_wordle_solver():
    """Main game loop for the Wordle solver."""
    print("=== WORDLE SOLVER ===")
    print("This program will help you solve Wordle puzzles.")
    print("For each guess, enter the feedback from Wordle:")
    print("  0 = Letter not in word")
    print("  1 = Letter in word but wrong position")
    print("  2 = Letter in word and correct position")

    # Setup
    df_words = download_word_list()

    # Filter out words that have already been used in past Wordle puzzles
    df_words = filter_out_used_words(df_words)

    max_guesses = 6
    guess_count = 0

    # Game state
    known_positions = ['', '', '', '', '']
    right_letters = []
    wrong_positions = [[], [], [], [], []]
    wrong_letters = []

    # Game loop
    while guess_count < max_guesses:
        remaining_words = filter_words(df_words, known_positions, right_letters, wrong_positions, wrong_letters)
        word_count = len(remaining_words)

        print(f"\n=== GUESS {guess_count + 1}/{max_guesses} ===")
        display_game_state(known_positions, right_letters, wrong_positions, wrong_letters)
        print(f"\nPossible words remaining: {word_count}")

        if word_count == 0:
            print("No words match the current constraints. Please check your inputs.")
            # Option to correct previous input
            continue

        if word_count <= 10:
            print("Possible words:")
            for word in remaining_words['word'].head(10).tolist():
                print(f"  {word}")

        # Show top words and letters if there are many possibilities
        if word_count > 1:
            if word_count > 10:
                visualize_top_words(remaining_words)

            df_letters = calculate_letter_probabilities(remaining_words)
            visualize_letter_probabilities(df_letters)
        elif word_count == 1:
            print(f"The solution is: {remaining_words['word'].iloc[0]}")
            break

        # Get user input
        word_guess = input("\nEnter your guess (5 letters): ").lower()
        while len(word_guess) != 5 or not word_guess.isalpha():
            word_guess = input("Please enter a valid 5-letter word: ").lower()

        feedback = input(f"Enter feedback for '{word_guess}' (e.g., 01202): ")
        while len(feedback) != 5 or not all(c in '012' for c in feedback):
            feedback = input("Please enter 5 digits (0, 1, or 2): ")

        # Check if word is solved
        if feedback == '22222':
            print(f"\nCongratulations! The word is '{word_guess}'")
            break

        # Process the guess
        process_guess(word_guess, feedback, known_positions, right_letters, wrong_positions, wrong_letters)
        guess_count += 1

    if guess_count == max_guesses and feedback != '22222':
        print("\nYou've used all your guesses. Better luck next time!")


if __name__ == "__main__":
    # Choose which mode to run
    run_wordle_scraper()
    run_wordle_solver()

