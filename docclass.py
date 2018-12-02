#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
import re

# Тест
def getwords(doc):
    splitter = re.compile(r'\W*', re.U)
    # Разбить на слова по небуквенным символам
    decodableText = splitter.split(doc)
    words = [s.lower() for (s) in decodableText
             if len(s) > 2 and len(s) < 20]
    # Вернуть набор уникальных слов
    return dict([(w, 1) for w in words])

def sampletrain(cl):
    cl.train('в 13-м туре бундеслиги бавария отправляется в гости к вердеру, дортмундская боруссия принимает фрайбург, а рб рейпциг сыграет с боруссией из менхенгладбаха.', 'футбол')
    cl.train('я не думаю, что при такой погоде есть смысл говорить об игре, потому что это не футбол, а лотерея. наверное, те, кто принимали решение о проведении матча, хотели провести игру независимо от того, что будет дальше. остаюсь при своем мнении. сегодняшнюю игру называть футболом нельзя.', 'футбол')
    cl.train('Нападающий «Сельты» Яго Аспас оформил дубль в матче с «Уэской» (2:0, второй тайм). На счету испанца 10 голов в этом сезоне. Он обогнал игроков «Барселоны» Лионеля Месси и Луиса Суареса, которые забили по 9 голов. Лидирует вместе с Аспасом в гонке бомбардиров Кристиан Стуани из «Жироны» – у него тоже 10 голов.', 'футбол')

    cl.train('зрада не заставила себя долго ждать: путин и трамп поприветствовали друг друга в кулуарах саммита. об этом сообщил пресс-секретарь российского президента дмитрий песков. он отметил, что президенты поздоровались, при этом возможности пообщаться у них не было.', 'политика')
    cl.train('видеоагентство ruptly опубликовало кадры с рабочего завтрака президента россии владимира путина и канцлера фрг ангелы меркель в аргентине на саммите G20. на записи видно, как лидеры стран обмениваются рукопожатиями в обеденном зале, после чего путин предлагает сесть.', 'политика')
    cl.train('Президент Украины Петр Порошенко заявил, что в тех областях страны, где было объявлено о введении военного положения, пройдут учебные сборы резервистов. Об этом политик рассказал 1 декабря во время передачи новой техники военным частям Киевской области.', 'политика')

    cl.train('Рецепт из говядины с заменой традиционных ингредиентов - это его версия, суп-харчо. Однако, также пикантный и вкусный. грудинка говяжья, лук репчатый, рис, помидоры, чеснок, перец душистый горошком, хмели-сунели, перец красный стручковый, перец красный молотый, соль, зелень кинзы.', 'рецепт')
    cl.train('Блины на молоке самые распространённые. Есть разные виды таких блинов и разные способы как сделать блины на молоке. Есть рецепт заварные блины на молоке, блины из прокисшего молока, толстые блины на молоке, блины на кефире и молоке, блины без яиц на молоке, блины на сухом молоке, дрожжевые блины на молоке и много других вариантов как делать блины на молоке.', 'рецепт')
    cl.train('В сотейнике разогреть масло, выложить чеснок и имбирь, готовить, помешивая, около 1 минуты, затем снять сотейник с плиты. Ввести в сотейник вино (бульон), мед, соевый соус, кунжут, цедру и острый соус, хорошо перемешать. В отдельную миску отлить 1/2 стакана маринада и отставить в сторону. Остальной маринад налить в большой пластиковый пакет с застежками, туда же выложить мясо, застегнуть пакет и хорошо встряхнуть. Положить пакет с мясо в холодильник на 1 час.', 'рецепт')

class classifier:
    def __init__(self, getfeatures, filename=None):
        # Счетчики комбинаций признак/категория
        self.fc = {}
        # Счетчики документов в каждой категории
        self.cc = {}
        self.getfeatures = getfeatures
        self.thresholds = {}

    # def __init__(self, getfeatures):
    #     classifier.__init__(self, getfeatures)
    #     self.thresholds = {}

    def setthreshold(self, cat, t):
        self.thresholds[cat] = t

    def getthreshold(self, cat):
        if cat not in self.thresholds: return 1.0
        return self.thresholds[cat]

    def classify(self, item, default = None, rightAnswer = None):
        probs = {}
        # Найти категорию с максимальной вероятностью
        max = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat

        # Убедиться, что найденная вероятность больше чем threshold*следующая по
        # величине
        for cat in probs:
            if cat == best: continue
            if probs[cat] * self.getthreshold(best) > probs[best]: return default

        if rightAnswer != best:
            return "Ошибка при классификации!!! Правильный ответ: " + rightAnswer + " Классификатор определил: " + best

        return "Классификатор сработал верно! Правильный ответ: " + rightAnswer + " Классификатор определил: " + best

    # Увеличить счетчик пар признак/категория
    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    # Увеличить счетчик применений категории
    def incc(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    # Сколько раз признак появлялся в данной категории
    def fcount(self,f,cat):
         if f in self.fc and cat in self.fc[f]:
           return float(self.fc[f][cat])
         return 0.0

    # Сколько образцов отнесено к данной категории
    def catcount(self,cat):
         if cat in self.cc:
           return float(self.cc[cat])
         return 0

    # Общее число образцов
    def totalcount(self):
         return sum(self.cc.values( ))

    # Список всех категорий
    def categories(self):
         return self.cc.keys( )


    def train(self, item, cat):
        features = (self.getfeatures(item))
        # Увеличить счетчики для каждого признака в данной классификации
        for f in features:
            self.incf(f, cat)
        # Увеличить счетчик применений этой классификации
        self.incc(cat)

    def fprob(self, f, cat):
        if self.catcount(cat) == 0: return 0
        # Общее число раз, когда данный признак появлялся в этой категории,
        # делим на количество образцов в той же категории
        return self.fcount(f, cat) / self.catcount(cat)

    def weightedprob(self, f, cat, prf, weight=1.0, ap=0.5):
        # Вычислить текущую вероятность
        basicprob = prf(f, cat)
        # Сколько раз этот признак встречался во всех категориях
        totals = sum([self.fcount(f, c) for c in self.categories()])
        # Вычислить средневзвешенное значение
        bp = ((weight * ap) + (totals * basicprob)) / (weight + totals)
        return bp

class bayes(classifier):
    def docprob(self, item, cat):
        features = self.getfeatures(item)

        # Перемножить вероятности всех признаков
        p = 1
        for f in features: p *= self.weightedprob(f, cat, self.fprob)
        return p

    def prob(self, item, cat):
        catprob = self.catcount(cat) / self.totalcount()
        docprob = self.docprob(item, cat)
        return docprob * catprob


