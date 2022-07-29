## Embedding stats

### CODEWOE
For words that were not found in Glove, 0 vector of same size was utilized
Train:
27292/31444 = 0.8679 in glove with 14699 unique values
4152/31444 = 0.1321 not in glove with 3494 unique values 

Dev:
4013/4593 = 0.8737 in glove with 2124 unique values
580/4593 = 0.1263 not in glove with 486 unique values

Test:
3893/4529 = 0.8596 in glove with 2157 unique values
636/4529 = 0.1404 not in glove with 539 unique values

For words that were not found in FastText, 0 vector of same size was utilized 300 Dimensions

Train:
30781/31444 = 0.9789 in fasttext with 17598 unique values 
663/31444 = 0.0211 in fasttext with 595 unique values

Dev:
4502/4593 = 0.9802 in glove with 2529 unique values
91/4593 = 0.0198 in glove with 81 unique values

Test:
4433/4529 = 0.9788 in glove with 2608 unique values
96/4529 = 0.0212 in glove with 88 unique values




FastText data already extracted...
------------------- Extracting FastText Embeddings -------------------
30574 corresponding words found in FastText out of 31444 words in our set: 2.766823559343595 percent and found 17393 unique words
870 corresponding words found in FastText out of 31444 words in our set: 2.766823559343595 percent and found 769 unique words


 Keys found in PKL data dictionary: dict_keys(['word', 'gloss', 'index', 'example', 'sgns', 'char', 'electra', 'glove', 'fasttext'])
------------------- Extracting FastText Embeddings -------------------
4475 corresponding words found in FastText out of 4593 words in our set: 2.5691269322882646 percent and found 2507 unique words
118 corresponding words found in FastText out of 4593 words in our set: 2.5691269322882646 percent and found 103 unique words


 Keys found in PKL data dictionary: dict_keys(['word', 'gloss', 'index', 'example', 'sgns', 'char', 'electra', 'glove', 'fasttext'])
------------------- Extracting FastText Embeddings -------------------
4396 corresponding words found in FastText out of 4529 words in our set: 2.936630602782071 percent and found 2573 unique words
133 corresponding words found in FastText out of 4529 words in our set: 2.936630602782071 percent and found 122 unique words