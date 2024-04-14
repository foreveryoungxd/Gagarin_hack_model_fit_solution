import typing as tp
import pandas as pd
import pickle
import re
from nltk import word_tokenize
from natasha import (
    Segmenter,
    NewsNERTagger,
    ORG,
    Doc,
    NewsEmbedding,
)
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import numpy as np

emb = NewsEmbedding()
segmenter = Segmenter()
ner_tagger = NewsNERTagger(emb)

EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[
    EntityScoreType
]  # list of entity scores,
#    for example, [(entity_id, entity_score) for entity_id, entity_score in entities_found]

def score_texts(
    messages: tp.Iterable[str], *args, **kwargs
) -> tp.Iterable[MessageResultType]:
    most_common_tokens = [
        'млрд',
        'млн',
        'трлн',
        'руб',
        'года',
        'компании',
        'компания',
        'компаний',
        'акции',
        'акций',
        'рублей',
        'год',
        'на',
        'году',
        'по',
        'сша',
        'результаты',
        'мсфо',
        'нефть',
        'нефти',
        'ru',
        'директоров',
        'Директоров'
        'при',
        'система',
        '30мск',
        '00мск',
        'совет',
        'акционеров',
         'ak47pfl',
        'если',
        'вопроc',
        'новости',
        'подробнее',
        'российских',
        'рамках',
        'ipo',
        'взгляд',
        'рф',
        'россии',
        'россия',
        'россие',
        'не',
        'дня',
        'дивиденды',
        'отчетность',
        'отчётность',
        'биржа',
        'бирже',
        'биржи',
        'кв',
        'мск',
        'гг',
        'фн',
        'upd'
    ]

    not_company_name_natasha = [
        'лонг',
        'мира',
        'горячую',
        'казначейство',
        'распадской',
        'детского',
        'ведомости',
        'акционеры',
        'шорт',
        'десятку',
        'акция'
        'цб',
        'мосбиржи',
        'мосбиржа',
        'банка',
        'сд',
        'компания',
        'совет',
        'система',
        'прайм',
        'ао',
        'рао',
        'московской',
        'тасс',
        'фрс',
        'минфин',
        'группы',
        'биржа',
        'бирже',
        'биржи',
        'en',
        'фас',
        'интерфакс',
        'за',
        'выручка',
        'не',
        'менеджмент',
        'инвестиции',
        'инвестиций',
        'сми',
        'флэт',
        'фонд',
        'топ',
        'новости',
        'боковик',
        'минфина',
        'системы',
        'центр',
        'фн',
        'сегодня',
        'рекомендовал',
        'прогноз',
        'сигналов',
        'от',
        'размере',
        'риа',
        'объем',
        'аутсайдеры',
        'отчет',
        'правления',
        'московский',
        'служба',
        'технический',
        'приказ',
        'интерфакса',
        'распадской',
        'арбитражный',
        'до'
    ]

    russian_stopwords = ['и',
     'в',
     'во',
     'не',
     'что',
     'он',
     'на',
     'я',
     'с',
     'со',
     'как',
     'а',
     'то',
     'все',
     'она',
     'так',
     'его',
     'но',
     'да',
     'ты',
     'к',
     'у',
     'же',
     'вы',
     'за',
     'бы',
     'по',
     'только',
     'ее',
     'мне',
     'было',
     'вот',
     'от',
     'меня',
     'еще',
     'нет',
     'о',
     'из',
     'ему',
     'теперь',
     'когда',
     'даже',
     'ну',
     'вдруг',
     'ли',
     'если',
     'уже',
     'или',
     'ни',
     'быть',
     'был',
     'него',
     'до',
     'вас',
     'нибудь',
     'опять',
     'уж',
     'вам',
     'ведь',
     'там',
     'потом',
     'себя',
     'ничего',
     'ей',
     'может',
     'они',
     'тут',
     'где',
     'есть',
     'надо',
     'ней',
     'для',
     'мы',
     'тебя',
     'их',
     'чем',
     'была',
     'сам',
     'чтоб',
     'без',
     'будто',
     'чего',
     'раз',
     'тоже',
     'себе',
     'под',
     'будет',
     'ж',
     'тогда',
     'кто',
     'этот',
     'того',
     'потому',
     'этого',
     'какой',
     'совсем',
     'ним',
     'здесь',
     'этом',
     'один',
     'почти',
     'мой',
     'тем',
     'чтобы',
     'нее',
     'сейчас',
     'были',
     'куда',
     'зачем',
     'всех',
     'никогда',
     'можно',
     'при',
     'наконец',
     'два',
     'об',
     'другой',
     'хоть',
     'после',
     'над',
     'больше',
     'тот',
     'через',
     'эти',
     'нас',
     'про',
     'всего',
     'них',
     'какая',
     'много',
     'разве',
     'три',
     'эту',
     'моя',
     'впрочем',
     'хорошо',
     'свою',
     'этой',
     'перед',
     'иногда',
     'лучше',
     'чуть',
     'том',
     'нельзя',
     'такой',
     'им',
     'более',
     'всегда',
     'конечно',
     'всю',
     'между',
     'это',
     'как',
     'так',
     'и',
     'в',
     'над',
     'к',
     'до',
     'не',
     'на',
     'но',
     'за',
     'то',
     'с',
     'ли',
     'а',
     'во',
     'от',
     'со',
     'для',
     'о',
     'же',
     'ну',
     'вы',
     'бы',
     'что',
     'кто',
     'он',
     'она',
     'оно',
     'из-за',
     'также'
     ]

    with open('latin_companies_mentions.pickle', 'rb') as f:
        latin_comps_mentions = pickle.load(f)

    with open('all_company_names_dict.pickle', 'rb') as f:
        all_names_dict = pickle.load(f)

    model_tfidf = load('TFIDF_fitted_model.joblib')


    df = pd.DataFrame(messages, columns=['MessageText'])

    def message_text_preprocess(text):  # Функция для предобработки текста
        remove_punctuation = '!"#$%&\'*+,./:;<=>?@[\\]^_`{|}~``🇷🇺'')('
        text = re.sub(r'\s+', ' ',
                      text)  # Заменяем последовательности пробелов и других пробельных символов на один пробел
        text = re.sub(r'(?<=[^\w\d])-|-(?=[^\w\d])|[^\w\d\s-]', '',
                      text)  # Удаляем все символы, кроме букв, цифр, пробелов и дефисов
        text = re.sub(r'\+\d{1,2}\s\(\d{3}\)\s\d{3}-\d{2}-\d{2}', '', text)  # удаляем телефонные номера
        text = re.sub(r'https?://\S+', '', text)  # удаляем ссылки
        text = re.sub(r'\s+', ' ', text)  # еще раз удаляем пробелы, если таковые остались
        text = re.sub('•', '', text)
        text = re.sub("''", '', text)
        text = re.sub(r'[«»]', '', text)
        text = re.sub(r'\d+', '', text)  # удаляем все цифры
        text = re.sub(r'\b\w\b', '', text)  # удаляем одиночные символы

        tokens = word_tokenize(text)  # Токенизируем текст
        filtered_tokens = [token for token in tokens if
                           token not in remove_punctuation and token not in russian_stopwords and token.lower() not in latin_comps_mentions and token.lower() not in most_common_tokens]

        return " ".join(filtered_tokens)

    df['MessageText'] = df['MessageText'].apply(message_text_preprocess)

    def fill_natasha_mentions(text):
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        orgs = []
        for item in doc.spans:
            if len(orgs) >= 3:  # Если уже найдено 3 организации, прекращаем извлечение
                break
            if item.type == 'ORG' and item.text.lower() not in not_company_name_natasha:
                orgs.append(item.text[:10])
        return orgs

    df['natasha_mentions'] = df['MessageText'].apply(fill_natasha_mentions)

    def find_best_companies(model_tfidf, mentions,
                            all_names_dict):  # функция для алгоритмического предсказания упоминания компании в тексте
        if len(mentions) == 0:  # для случая отсутствия выявленных сущностей в тексте
            return 0
        natasha_vector = model_tfidf.transform(
            [' '.join(mentions)])  # Преобразуем упоминания Natasha и компаний в вектора TF-IDF
        company_vectors = model_tfidf.transform([' '.join(company_names) for company_names in all_names_dict.values()])

        similarities = cosine_similarity(natasha_vector,
                                         company_vectors)  # Находим косинусное расстояние между упоминаниями и компаниями

        best_company_id = similarities.argsort(axis=1)[:, -1] + 1  # компания с наивысшим скором
        second_best_company_id = similarities.argsort(axis=1)[:, -2] + 1  # компания со вторым по величине скором

        if similarities.max(axis=1)[0] - np.partition(similarities, -2, axis=1)[:, -2][
            0] > 0.05:  # условие разницы топ-1 и топ-2 скора
            return best_company_id[0]
        else:
            return [best_company_id[0], second_best_company_id[0]]

    df['predict_id'] = df['natasha_mentions'].apply(
        lambda x: find_best_companies(model_tfidf, x, all_names_dict))

    df = df.explode('predict_id')

    to_predict_tfidf = model_tfidf.transform(df['MessageText'].values)

    clf = load('RF_clf.joblib')

    df['predict_sentiment'] = clf.predict(to_predict_tfidf)

    grouped = df.groupby('MessageText')

    def transform_group(group): #Функция для преобразования каждой группы в нужный формат
        pairs = []
        for _, row in group.iterrows():
            if row['predict_id'] == 0:
                pairs.append(())
            else:
                pairs.append((row['predict_id'], row['predict_sentiment']))
        return [pairs]

    result = [transform_group(group) for _, group in grouped] #Преобразование каждой группы и объединение результатов в список

    return result


    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)

    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages returns MessageResultType object
    -------
    Clarifications:
    >>> assert all([len(m) < 10 ** 11 for m in messages]) # all messages are shorter than 2048 characters
    """
    raise NotImplementedError
