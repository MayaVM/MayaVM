from Algorithms import *
from DataLoader import *
from Eval import *

### Generting the data:
# Synth
Synth_A, Synth_b = gen_cluster(1000, 2000, 5)

# USPS

USPS_A, USPS_b = h5_loader('USPS/usps.h5')


# coil
coil_A, coil_b = img_loader('coil-20-proc', 4)

# orl
ORL_A, ORL_b = img_loader ('ORL', 1)

# WarNews
war_news_A = text_loader('war_news/war-news.csv', 'Headlines')
war_news_b = lable_loader('war_news/war-news.csv', 'Keyword')

# PartOfSpeech
PartOfSpeech_A = text_loader('words-by-partsOfSpeech/data.csv', 'Word')
PartOfSpeech_b = lable_loader('words-by-partsOfSpeech/data.csv', 'PartsOfSpeech')



### Setting parameters for all test runs:

alg_set = [SampleSVD, SamplApproxSVD, RP, SVD, ApprSVD, elkan, k_means]
alg_names = ['SampleSVD', 'SamplApproxSVD', 'RP', 'SVD', 'ApprSVD', 'EM', 'kMeans']
data_set = [Synth_A, USPS_A, coil_A, ORL_A, war_news_A, PartOfSpeech_A]
data_names = ['Synth_A', 'USPS_A', 'coil_A', 'ORL_A', 'war_news_A', 'PartOfSpeech_A']
data_val = [Synth_b, USPS_b, coil_b, ORL_b, war_news_b, PartOfSpeech_b]
dim = [i for i in range(5, 105, 5)]

tests = ['time', 'objective value', 'accuracy', 'accuracy2']

plt_data(data_set, data_names, data_val, alg_set, alg_names, dim)
