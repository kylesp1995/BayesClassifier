#!/usr/bin/env python
# -*- coding: utf-8 -*-

import docclass
import pymorphy2
import json
from supportFunction import mergeDict, readDictFromFile, writeDictToFile, mergeNestedDict
import datetime
from os import listdir
from os.path import isfile, join

withTrain = False

rightAnswerFootball = 'спорт'
rightAnswerPolitic = 'политика'
rightAnswerRecept = 'рецепт'
unknownAnswer = 'unknown'
mypath = '/Users/mihailageev/BayesClassifier/text_for_detect'


# cl=docclass.bayes(docclass.getwords)
# if withTrain:
#     docclass.sampletrain(cl)


def test_func(cl):
    mypath = '/Users/mihailageev/BayesClassifier/text_for_detect'
    unknownAnswer = 'unknown'

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    print(datetime.datetime.now())

    for file in onlyfiles:
        f = open(mypath + '/' + file, encoding="utf8", errors='ignore')
        text = f.read()

        file = file.split('_')
        title = (file[0])
        print(cl.classify(withTrain, text, default=unknownAnswer, rightAnswer=title))
        file.clear()

    print(datetime.datetime.now())


cl = docclass.bayes(docclass.getwords)
if withTrain:
    docclass.sampletrain(cl)

test_func(cl)

# cl=docclass.bayes(docclass.getwords)
# if withTrain:
#     docclass.sampletrain(cl)
# print(datetime.datetime.now())
#
#
#
# print ("\n")
#
# print (cl.classify(withTrain,'комментарий по матчу? еще раз: это лотерея. вы видели, как залетел мяч, сколько было брака, какие действия были у игроков на поле. в такую погоду и мяч, и действия некоторых игроков были деревянными.', default=unknownAnswer, rightAnswer=rightAnswerFootball))
# print (cl.classify(withTrain,'Нападающий «Сельты» Яго Аспас оформил дубль в матче с «Уэской» (2:0, второй тайм). На счету испанца 10 голов в этом сезоне. Он обогнал игроков «Барселоны» Лионеля Месси и Луиса Суареса, которые забили по 9 голов. Лидирует вместе с Аспасом в гонке бомбардиров Кристиан Стуани из «Жироны» – у него тоже 10 голов.', default=unknownAnswer, rightAnswer=rightAnswerFootball))
# print (cl.classify(withTrain,'УЕФА запустил голосование за лучшего игрока недели в Лиге чемпионов. На награду претендуют Лионель Месси («Барселона»), забивший гол и сделавший ассист в игре против ПСВ (2:1), Арьен Роббен («Бавария»), сделавший дубль в ворота «Бенфики» (5:1), Дрис Мертенс («Наполи»), забивший 2 мяча в ворота «Црвены Звезды» (3:1), а также Максвелл Корне («Лион»), который сделал дубль в матче с «Манчестер Сити» (2:2).', default=unknownAnswer, rightAnswer=rightAnswerFootball))
# print (cl.classify(withTrain,'Павел Садырин пришел в «Зенит» в 1965 году и вскоре стал любимцем ленинградской публики. Он выделялся неуступчивостью в единоборствах и убойным ударом. Выступал за сине-бело-голубых 11 сезонов подряд, сыграв в 333 матчах. В 1970 году Садырин был избран капитаном и до окончания игровой карьеры выводил команду на поле. «„Зенит“ был бы идеальной командой, если бы в нем играло 11 Садыриных», — писали тогда газеты.', default=unknownAnswer, rightAnswer=rightAnswerFootball))
# #
# # print ("\n\n\n")
#
# print (cl.classify(withTrain,'трамп обсудил важные вопросы с президентом россии',default=unknownAnswer, rightAnswer=rightAnswerPolitic))
# print (cl.classify(withTrain,'трамп и путин приехали на саммит',default=unknownAnswer, rightAnswer=rightAnswerPolitic))
# print (cl.classify(withTrain,'видеоагентство ruptly опубликовало кадры с рабочего завтрака президента россии владимира путина и канцлера фрг ангелы меркель в аргентине на саммите G20. на записи видно, как лидеры стран обмениваются рукопожатиями в обеденном зале, после чего путин предлагает сесть.', default=unknownAnswer, rightAnswer=rightAnswerPolitic))
# print (cl.classify(withTrain,'Президенты России и США Владимир Путин и Дональд Трамп на саммите "двадцатки" не общались, но поприветствовали друг друга. Об этом ТАСС сообщил пресс-секретарь главы государства Дмитрий Песков, отвечая на вопрос, удалось ли лидерам перекинуться парой слов.', default=unknownAnswer, rightAnswer=rightAnswerPolitic))
# #
# # print ("\n\n\n")
# #
# # print (cl.classify(withTrain,'Сварить бульон. Картофель нарезать ломтиками и опустить в кипящий и подсоленный бульон. Пока варится картофель, нашинковать капусту, лук, морковь, помидоры (чеснок по желанию). Припустить овощи для щей в масле на сковороде, добавить порезанные помидоры(очищенные от шкурки) и мелко порубленный чеснок, тушить 5 минут.', default=unknownAnswer, rightAnswer=rightAnswerRecept))
# # print  (cl.classify(withTrain,'Торт из омлета нечасто встречается на наших столах, и совсем напрасно. Яичные блинчики можно перемазать любой начинкой, на усмотрение хозяйки, украсить - и любители омлета будут в восторге! Подайте омлет в виде закусочного торта - и такая холодная закуска станет выигрышным вариантом и для праздничного меню, и для домашнего ужина.', default=unknownAnswer, rightAnswer=rightAnswerRecept))
# #
# # print ("\n\n\n")
# #
# print (cl.classify(withTrain,'Совет нацбезопасности и обороны Украины утвердил решение о введении новых санкций против России, сообщила его пресс-служба.',default=unknownAnswer, rightAnswer=rightAnswerPolitic))
# print (cl.classify(withTrain,'В российском МИД назвали авиаудары, нанесенные израильской авиацией по сирийской территории, грубым нарушением суверенитета арабской республики. При этом в Минобороны обвинили Израиль в создании угрозы для пассажирских самолетов.',default=unknownAnswer, rightAnswer=rightAnswerPolitic))
# print (cl.classify(withTrain,'Президент Владимир Путин на традиционной предновогодней встрече с предпринимателями заявил, что видит большую роль крупного бизнеса в реализации национальных проектов. Нацпроекты — важнейшее направление работы правительства на ближайшие годы, отметил он.',default=unknownAnswer, rightAnswer=rightAnswerPolitic))
# print (cl.classify(withTrain,'Секретарь Совета безопасности России Николай Патрушев заявил, что в Европе формируется неонацистский союз с опорой на ультраправые силы. Об этом Патрушев рассказал в интервью «Российской газете», опубликованном сегодня, 26 декабря.',default=unknownAnswer, rightAnswer=rightAnswerPolitic))
