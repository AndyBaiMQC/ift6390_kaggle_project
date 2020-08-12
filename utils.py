import re
import string

puncts = ['☹', '＞', 'ξ', 'ட', '「',  '½', '△', 'É', '¿', 'ł',
          '¼', '∆', '≥', '⇒', '¬', '∨', 'č', 'š', '∫', '▾', 'Ω', 
          '＾', 'ý', 'µ', '?', '!', '.', ',', '"', '#', '$', '%',
          '\\', "'", '(', ')', '*', '+', '-', '/', ':', ';', '<',
          '=', '>', '@', '[', ']', '^', '_', '`', '{', '|', '}', 
          '~', '“', '”', '’', '′', '…', 'ɾ', '̃', 'ɖ', '–', '‘',
          '√', '→',  '—', '£', 'ø', '´', '×', 'í', '÷', 'ʿ', '€',
          'ñ', 'ç', 'へ', '↑', '∞', 'ʻ', '℅''ι', '•', 'ì', '−', '∈',
          '∩', '⊆', '≠', '∂', 'आ', 'ह', 'भ', 'ी', '³', 'च', '...', 
          '⌚', '⟨', '⟩', '∖', '˂',  '☺', 'ℇ', '❤', '♨', '✌', 'ﬁ', 
          'て', '„', '¸', 'ч',  '⧼', '⧽', 'ম', 'হ', 'ῥ', 'ζ', 'ὤ',
          'Ü', 'Δ',  'ʃ', 'ɸ', 'ợ', 'ĺ', 'º', 'ष', '♭', '़', '✅', 
          '✓', '∘', '¨', '″', 'İ', '⃗', '̂', 'æ', 'ɔ', '∑', '¾', '≅',
          '‑', 'ֿ','ő', '－', 'ș', 'ן', 'Γ', '∪', '⊨', '∠', 'Ó', '«', 
          '»', 'Í', 'க', 'வ', 'ா', 'ம', '≈','،', '＝', '（', '）', 'ə',
          'ਨ', 'ਾ', 'ਮ', 'ੁ', '︠', '︡', 'ː', '∧', '∀', 'Ō', 'ㅜ', 
          'ण', '≡',  '《', '》', 'ٌ', 'Ä', '」']
          
puncts = list(set(puncts[:]))

def clean_symbol(text):
    for s in puncts:
        text = text.replace(s, f' {s} ')
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub('\n', ' ', text)
    text = re.sub("'ll", ' will', text)
    text = re.sub("'ve", ' have', text)
    text = re.sub("'s", ' is', text)
    text = re.sub("'d", ' would', text)
    text = re.sub("'m", ' am', text)
    text = re.sub("'re", ' are', text)
    text = re.sub("n't", ' not', text)
    text = re.sub("won't", 'will not', text)
    text = re.sub('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'ip', text)
    return text

def clean_url(text):
    text = re.sub('((www.[^\s]+)|(https?://[^\s]+))','url',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

def character_range(text): 
    for ch in string.ascii_lowercase[:27]:
        if ch in text:
            template = r"(" + ch + ")\\1{2,}"
            text = re.sub(template, ch, text)
    return text