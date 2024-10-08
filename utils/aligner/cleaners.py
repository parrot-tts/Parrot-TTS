import re
import inflect
from unidecode import unidecode

_inflect = inflect.engine()

_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('&', 'and'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
    ('tts', 'text to speech'),
]]

hindi_numbers = {
    0: 'शून्य', 1: 'एक', 2: 'दो', 3: 'तीन', 4: 'चार', 5: 'पाँच', 6: 'छह', 7: 'सात', 8: 'आठ', 9: 'नौ',
    10: 'दस', 11: 'ग्यारह', 12: 'बारह', 13: 'तेरह', 14: 'चौदह', 15: 'पंद्रह', 16: 'सोलह', 17: 'सत्रह', 18: 'अठारह', 19: 'उन्नीस',
    20: 'बीस', 30: 'तीस', 40: 'चालीस', 50: 'पचास', 60: 'साठ', 70: 'सत्तर', 80: 'अस्सी', 90: 'नब्बे',
    100: 'सौ', 200: 'दो सौ', 300: 'तीन सौ', 400: 'चार सौ', 500: 'पाँच सौ', 600: 'छह सौ', 700: 'सात सौ', 800: 'आठ सौ', 900: 'नौ सौ'
}

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text

def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def expand_numbers(text):
    return normalize_numbers(text)

def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(
                num, andword="", zero="oh", group=2
            ).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")
    
def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def english_cleaners(input_text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    input_text = unidecode(input_text)
    input_text = input_text.lower()
    input_text = expand_numbers(input_text)
    input_text = expand_abbreviations(input_text)
    input_text = collapse_whitespace(input_text)
     
    # Define unwanted characters to remove
    unwanted_chars = ['#','+','\\', '_', '`', '@', '/', '-', "'",'>','<', '(', ')', '*', '"', ':', ';', '!']
    
    # Remove non-printable ASCII characters
    cleaned_text = re.sub(r'[^\x20-\x7E]', '', input_text)

    # Remove specific unwanted characters
    for char in unwanted_chars:
        cleaned_text = cleaned_text.replace(char, '')
    
    cleaned_text = cleaned_text.replace('&', 'and')

    return cleaned_text

def nonenglish_cleaners(input_text):
    '''Pipeline for Non English text, including number and abbreviation expansion.'''
    input_text = unidecode(input_text)
    input_text = input_text.lower()
    input_text = collapse_whitespace(input_text)
     
    for char in ['0','1','2','3','4','5','6','7','8','9']:
        input_text=input_text.replace(str(char),'')
    
    # Define unwanted characters to remove
    unwanted_chars = ['|','%','+','=','[',']','^','\\','{','}', '_', '`', '‘', '’', '@', '/', '-', "'",'>','<', '(', ')', '*', '"', ':', ';', '!']
    
    # Remove non-printable ASCII characters
    cleaned_text = re.sub(r'[^\x20-\x7E]', '', input_text)

    # Remove specific unwanted characters
    for char in unwanted_chars:
        cleaned_text = cleaned_text.replace(char, '')
    
    cleaned_text = cleaned_text.replace('&', 'and')
    
    # Remove extra spaces
    cleaned_text = ' '.join(cleaned_text.split())

    return cleaned_text

def number_to_hindi(num):
    if num in hindi_numbers:
        return hindi_numbers[num]
    
    if num < 100:  # For numbers between 21-99
        tens = (num // 10) * 10
        ones = num % 10
        return hindi_numbers[tens] + ' ' + hindi_numbers[ones]
    
    if num < 1000:  # For numbers between 101-999
        hundreds = (num // 100) * 100
        remainder = num % 100
        if remainder == 0:
            return hindi_numbers[hundreds]
        else:
            return hindi_numbers[hundreds] + ' ' + number_to_hindi(remainder)
        
def replace_devanagari_numbers(text):
    # Pattern to find numbers in Devanagari
    devanagari_num_pattern = r'[०१२३४५६७८९]+'
    
    # Function to convert Devanagari number string to an integer
    def devanagari_to_int(dev_num):
        devanagari_numerals = {'०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'}
        num_str = ''.join([devanagari_numerals[c] for c in dev_num])
        return int(num_str)

    # Replace each Devanagari number in the text
    def replace_match(match):
        devanagari_num = match.group(0)
        number = devanagari_to_int(devanagari_num)
        return number_to_hindi(number)

    # Substitute Devanagari numbers with their Hindi word equivalents
    return re.sub(devanagari_num_pattern, replace_match, text)

def nonenglish_cleaners_no_transliteration(input_text):
    '''Pipeline for Non English text, with no transliterations'''
    input_text = collapse_whitespace(input_text)
     
    for char in ['0','1','2','3','4','5','6','7','8','9']:
        input_text=input_text.replace(str(char),'')
    
    # Define unwanted characters to remove
    unwanted_chars = ['—','–','…','“', '”','%','+','=','[',']','^','\\','{','}', '_', '`', '‘', '’', '@', '/', '-', "'",'>','<', '(', ')', '*', '"', ':', ';', '!']

    # Remove specific unwanted characters
    for char in unwanted_chars:
        input_text = input_text.replace(char, '')
        
    input_text = input_text.replace('|','.')

    # Replace numerals
    input_text = replace_devanagari_numbers(input_text)

    input_text = input_text.replace("\x92", "'")
    input_text = input_text.replace("\xad", "")

    # Remove extra spaces
    input_text = ' '.join(input_text.split())

    return input_text