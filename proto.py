import pandas as pd
from transformers import pipeline

from jpap.ipl import DescExtractor
from jpap import preprocessing as jpp


models = {"joeddav": "joeddav/xlm-roberta-large-xnli", 
          "vicgalle": "vicgalle/xlm-roberta-large-xnli-anli", # very big
          "MoritzLaurer" : "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
          "MoritzLaurer-mini" : "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli" # very fast => check if swedish works
          }
classifier = pipeline("zero-shot-classification", model = models["MoritzLaurer-mini"])

# examples:
df = pd.read_csv("data/created/industry_train.csv")
extractor = DescExtractor(postings = df["job_description"])

text = extractor.postings[101]
text = extractor.postings[130]

tokenized_text = extractor.tokenized_input[130]
targets = ["who we are", "who this is", "industry or sector"]
#targets = ["employer"]
labels = targets + ["address", "benefits", "other"]

employer_sentences = [classifier(s, candidate_labels = labels) for s in tokenized_text]
employer_desc = " ".join([s["sequence"] for s in employer_sentences if s["labels"][0] in targets])
employer_desc
len(employer_desc.split("."))

#### Out of vocabulary languages: --------------------

### xnli languages:
# English
# French
# Spanish
# German
# Russian
# Turkish
# Chinese
# Hindi

# Greek
# Bulgarian
# Arabic
# Vietnamese
# Thai
# Swahili
# Urdu

# missing: ---
# Danish  -> good
# Swedish -> good
# Dutch -> good
# Italian -> good
# Hebrew -> ??? scheint etwas schwieriger zu sein...
# Japanese -> weicht etwas ab, aber ok.
# Korean -> weicht etwas ab, aber ok.
# Persian -> ??? scheint etwas schwieriger zu sein...

import nltk
text = """
 برای مشتریان خصوصی، آنها را در تماس مستقیم با مشتری با مشاوره شایسته خود و اشتیاق خود به خدمات و محصولات هوشمند ما متقاعد خواهید کرد. چه آنلاین، چه از طریق تلفن یا حضوری: با تخصص و مشتری مداری خود، کارهای پیچیده را برای مشتریان ما آسان می کنید. این چیزی است که می توانید به آن دست یابید - شما به مشتریان خود در مورد مسائل مالی در شعبه، از طریق تلفن یا آنلاین مشاوره می دهید - شما مشترکاً مسئول ارائه خدمات و محصولات PostFinance در زمینه های سرمایه گذاری، تامین حقوق بازنشستگی و تامین مالی هستید - شما در اشتیاق خود برای خدمات و محصولات هوشمند PostFinance و قرار ملاقات های مشتری خود را با سهولت ترتیب دهید - با انگیزه و درجه بالایی از استقلال به اهداف فروش و توسعه توافق شده دست یابید و در موفقیت پست فاینانس آنچه با خود به ارمغان می آورید سهم مرتبطی داشته باشید - کارمند تجاری آموزش EFZ یا فروش، آموزش بیشتر در بخش مالی (به عنوان مثال مشاور مالی IAF) مورد نظر است - چند سال
تجربه در مشاوره شخصی مشتریان در بخش بانکی یا در بخش بیمه، تجربه در کسب و کار سرمایه گذاری و بازنشستگی - شخصیتی با انگیزه و مستقل با ویژگی های انجام دهنده و اشتیاق به خدمات و محصولات PostFinance - دانش بی عیب و نقص آلمانی، گفتاری و نوشتاری و همچنین دانش ایتالیایی مطلوب است، هر زبان خارجی دیگری مزیت پست فایننس است

"""
tokenized_text = nltk.sent_tokenize(text)

targets = ["who we are", "who this is", "industry or sector"]
labels = targets + ["address", "benefits", "other"]

employer_sentences = [classifier(s, candidate_labels = labels) for s in tokenized_text]
employer_desc = " ".join([s["sequence"] for s in employer_sentences if s["labels"][0] in targets])
employer_desc
len(employer_desc)
len(text)