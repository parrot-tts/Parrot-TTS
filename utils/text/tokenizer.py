from utils.text import cmudict, english
from string import punctuation
from g2p_en import G2p
import re


class ArpabetTokenizer: 
    pad = "<pad>"
    sep = "<sep>"
    arpabet = cmudict.valid_symbols
    silences = ['sp', 'spn', 'sil']
    _punctuation = "!'(),.:;? "
    
    def __init__(self):
        self.symbols = ([self.pad, self.sep]+ self.arpabet+self.silences+list(self._punctuation))

        self.stoi = {s: i for (i,s) in enumerate(self.symbols)}
        self.itos = {i: s for (i,s) in enumerate(self.symbols)}
        self.pad_idx = self.stoi[self.pad]
        self.sep_idx = self.stoi[self.sep]

    def __len__(self):
        return len(self.symbols)

    def tokenize(self, phoneme_seq):
        # only takes in preprocessed arpabet sequences
        return [self.stoi[s] for s in phoneme_seq]
    

class ArpabetPhonemizer:
    def __init__(self, lexicon_path):
        self.lexicon = cmudict.read_lexicon(lexicon_path)
        self.g2p = G2p()
        self.cleaner = english.english_cleaners
        self._curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")
        
    def word_to_phoneme(self, word):
        if word in punctuation:
            return ['sp']
        if word in self.lexicon:
            return self.lexicon[word]
        else:
            return list(filter(lambda p: p != " ", self.g2p(word)))
        
    def phonemize(self, txt):
        phones = []
        # check for {}
        curly_match = self._curly_re.match(txt)
        if not curly_match:
            txt = self.cleaner(txt)
            words = re.split(r"([,;.\-\?\!\s+])", txt)
            for word in words:
                phones.extend(self.word_to_phoneme(word))
        else:
            parts = re.split(r'(\{.*?\})', txt)
            for part in parts:
                if part.startswith('{') and part.endswith('}'):
                    phones.extend(part[1:-1].split())
                else:
                    clean_part = self.cleaner(part)
                    print(clean_part)
                    words = re.split(r"([,;.\-\?\!\s+])", clean_part)
                    print(words)
                    for word in words:
                        if word not in ('', ' '):
                            phones.extend(self.word_to_phoneme(word.lstrip().rstrip()))
        
        return phones