import operator
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt

df = pd.read_table('sms-spam-corpus.csv', sep='\t', names=['label', 'sms'])

df.sms = df.sms.str.replace(r'\d+', '', regex=True)
df.sms = df.sms.str.lower()

for ch in ["the", "a", "in", "to"]:
    df.sms = df.sms.str.replace(ch, '')
# pattern = r'[^A-Za-z0-9]+'
df.sms = df.sms.str.replace(r'[^a-zA-Z0-9 \n]', '', regex=True)

spam_sms_length = {}
ham_sms_length = {}
for index, row in df.iterrows():
    if row["label"] == "spam":
        if len(row["sms"]) in spam_sms_length:
            spam_sms_length[len(row["sms"])] += 1
        else:
            spam_sms_length[len(row["sms"])] = 1
    elif row["label"] == "ham":
        if len(row["sms"]) in ham_sms_length:
            ham_sms_length[len(row["sms"])] += 1
        else:
            ham_sms_length[len(row["sms"])] = 1

stemmer = SnowballStemmer("english")
df.sms = df.sms.str.split()
df.sms = df.sms.apply(lambda x: [stemmer.stem(y) for y in x])
count = 0

all_spam_words = []
all_ham_words = []
for index, row in df.iterrows():
    if row["label"] == "spam":
        all_spam_words.append(row["sms"])
    elif row["label"] == "ham":
        all_ham_words.append(row["sms"])

words_df = pd.DataFrame(
    [{"label": "spam", "words_list": all_spam_words}, {"label": "ham", "words_list": all_ham_words}])
spam_count_dict = {}
spam_length = {}
ham_count_dict = {}
ham_length = {}
for index, row in words_df.iterrows():
    if row["label"] == "spam":
        for arr in row["words_list"]:
            for word in arr:
                if word in spam_count_dict:
                    spam_count_dict[word] += 1
                else:
                    spam_count_dict[word] = 1

                if len(word) in spam_length:
                    spam_length[len(word)] += 1
                else:
                    spam_length[len(word)] = 1
    elif row["label"] == "ham":
        for arr in row["words_list"]:
            for word in arr:
                if word in ham_count_dict:
                    ham_count_dict[word] += 1
                else:
                    ham_count_dict[word] = 1

                if len(word) in ham_length:
                    ham_length[len(word)] += 1
                else:
                    ham_length[len(word)] = 1

spam_count = 0
for item in spam_count_dict:
    spam_count += spam_count_dict[item]

ham_count = 0
for item in ham_count_dict:
    ham_count += ham_count_dict[item]

for key, value in spam_length.items():
    num = spam_length[key]
    spam_length[key] = num / spam_count

for key, value in ham_length.items():
    num = ham_length[key]
    ham_length[key] = num / ham_count

sorted_spam_dict = sorted(spam_count_dict.items(), key=operator.itemgetter(1), reverse=True)
sorted_ham_dict = sorted(ham_count_dict.items(), key=operator.itemgetter(1), reverse=True)
sorted_word_length_spam = sorted(spam_length.items(), key=operator.itemgetter(0), reverse=True)
sorted_word_length_ham = sorted(ham_length.items(), key=operator.itemgetter(0), reverse=True)
sorted_sentence_length_spam = sorted(spam_sms_length.items(), key=operator.itemgetter(0), reverse=True)
sorted_sentence_length_ham = sorted(ham_sms_length.items(), key=operator.itemgetter(0), reverse=True)

print(spam_count_dict)
print(sorted_spam_dict)
print(spam_length)
print(sorted_sentence_length_ham)

with open("./output/spam_count.txt", 'w') as file:
    for item in spam_count_dict:
        file.writelines([str(item), ',', str(spam_count_dict[item]), '\n'])

with open("./output/ham_count.txt", 'w') as file:
    for item in ham_count_dict:
        file.writelines([str(item), ',', str(ham_count_dict[item]), '\n'])
# print(df)

def create_sub_curve(lengths, sub_curve_name, file_name):
    x = list()
    y = list()
    quantity = 0
    sum_for_median = 0
    for key, value in lengths:
        quantity += value
        sum_for_median += key * value
        x.append(key)
        y.append(value)
    mean = sum_for_median / quantity
    with open("./output/" + file_name, 'a+') as file:
        file.writelines(['mean ', str(sub_curve_name), ':', str(mean), '\n'])
    plt.plot(x, y, label=sub_curve_name)


def build_word_length_plot(spam_lengths, ham_lengths):
    create_sub_curve(spam_lengths, 'spam', 'word_length_mean.txt')
    create_sub_curve(ham_lengths, 'ham', 'word_length_mean.txt')
    plt.xlabel('words length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig("./output/words_frequency.png")
    plt.show()


def build_sentence_length_plot(spam_lengths, ham_lengths):
    create_sub_curve(spam_lengths, 'spam', 'sentence_length_mean.txt')
    create_sub_curve(ham_lengths, 'ham', 'sentence_length_mean.txt')
    plt.xlabel('sentence length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig("./output/sentence_frequency.png")
    plt.show()

def build_plot(content, file_name):
    i = 0
    words_list = list()
    nums = list()
    for k, v in content:
        if i == 13:
            break
        words_list.append(k)
        nums.append(v)
        i += 1
    plt.plot(words_list, nums, linewidth=2.0)
    plt.savefig("./output/" + file_name)
    plt.show()


build_plot(sorted_spam_dict, 'spam.png')
build_plot(sorted_ham_dict, 'ham.png')
build_word_length_plot(sorted_word_length_spam, sorted_word_length_ham)
build_sentence_length_plot(sorted_sentence_length_spam, sorted_sentence_length_ham)