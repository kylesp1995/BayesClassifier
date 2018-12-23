#!/usr/bin/env python
# -*- coding: utf-8 -*-

import docclass



# from os import listdir
# from os.path import isfile, join
#
# mypath = '/Users/mihailageev/BayesClassifier/train_text'
#
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#
# for file in onlyfiles:
#     text = open(mypath + '/' + file,encoding="utf8", errors='ignore')
#     print(text.read())
#     file = file.split('_')
#     print(file[0])
#
# print(onlyfiles)




rightAnswerFootball = 'футбол'
rightAnswerPolitic = 'политика'
rightAnswerRecept = 'рецепт'
unknownAnswer = 'unknown'

cl=docclass.bayes(docclass.getwords)
docclass.sampletrain(cl)
# for i in range(10): docclass.sampletrain(cl)

print ("\n")

print (cl.classify('озил играет за сборную германии',default=unknownAnswer, rightAnswer = 'футбол'))
print (cl.classify('комментарий по матчу? еще раз: это лотерея. вы видели, как залетел мяч, сколько было брака, какие действия были у игроков на поле. в такую погоду и мяч, и действия некоторых игроков были деревянными.', default=unknownAnswer, rightAnswer=rightAnswerFootball))
print (cl.classify('Нападающий «Сельты» Яго Аспас оформил дубль в матче с «Уэской» (2:0, второй тайм). На счету испанца 10 голов в этом сезоне. Он обогнал игроков «Барселоны» Лионеля Месси и Луиса Суареса, которые забили по 9 голов. Лидирует вместе с Аспасом в гонке бомбардиров Кристиан Стуани из «Жироны» – у него тоже 10 голов.', default=unknownAnswer, rightAnswer=rightAnswerFootball))
print (cl.classify('УЕФА запустил голосование за лучшего игрока недели в Лиге чемпионов. На награду претендуют Лионель Месси («Барселона»), забивший гол и сделавший ассист в игре против ПСВ (2:1), Арьен Роббен («Бавария»), сделавший дубль в ворота «Бенфики» (5:1), Дрис Мертенс («Наполи»), забивший 2 мяча в ворота «Црвены Звезды» (3:1), а также Максвелл Корне («Лион»), который сделал дубль в матче с «Манчестер Сити» (2:2).', default=unknownAnswer, rightAnswer=rightAnswerFootball))
print (cl.classify('Павел Садырин пришел в «Зенит» в 1965 году и вскоре стал любимцем ленинградской публики. Он выделялся неуступчивостью в единоборствах и убойным ударом. Выступал за сине-бело-голубых 11 сезонов подряд, сыграв в 333 матчах. В 1970 году Садырин был избран капитаном и до окончания игровой карьеры выводил команду на поле. «„Зенит“ был бы идеальной командой, если бы в нем играло 11 Садыриных», — писали тогда газеты.', default=unknownAnswer, rightAnswer=rightAnswerFootball))

print ("\n\n\n")

print (cl.classify('трамп обсудил важные вопросы с президентом россии',default=unknownAnswer, rightAnswer=rightAnswerPolitic))
print (cl.classify('трамп и путин приехали на саммит',default=unknownAnswer, rightAnswer=rightAnswerPolitic))
print (cl.classify('видеоагентство ruptly опубликовало кадры с рабочего завтрака президента россии владимира путина и канцлера фрг ангелы меркель в аргентине на саммите G20. на записи видно, как лидеры стран обмениваются рукопожатиями в обеденном зале, после чего путин предлагает сесть.', default=unknownAnswer, rightAnswer=rightAnswerPolitic))
print (cl.classify('Президенты России и США Владимир Путин и Дональд Трамп на саммите "двадцатки" не общались, но поприветствовали друг друга. Об этом ТАСС сообщил пресс-секретарь главы государства Дмитрий Песков, отвечая на вопрос, удалось ли лидерам перекинуться парой слов.', default=unknownAnswer, rightAnswer=rightAnswerPolitic))

print ("\n\n\n")

print (cl.classify('Каждую неделю мы выбираем несколько новых и не очень новых фильмов и сериалов, которые можно посмотреть на КиноПоиске. Тем, кто еще ни разу не пользовался сервисом, дарим промокод на первую покупку. При покупке нужно ввести слово DECEMBER, промокод актуален до конца месяца.', default=unknownAnswer, rightAnswer=unknownAnswer))
print (cl.classify('Сначала записывается часть if с условным выражением, далее могут следовать одна или более необязательных частей elif, и, наконец, необязательная часть else. Общая форма записи условной инструкции if выглядит следующим образом.', default=unknownAnswer, rightAnswer=unknownAnswer))

print ("\n\n\n")

print (cl.classify('Сварить бульон. Картофель нарезать ломтиками и опустить в кипящий и подсоленный бульон. Пока варится картофель, нашинковать капусту, лук, морковь, помидоры (чеснок по желанию). Припустить овощи для щей в масле на сковороде, добавить порезанные помидоры(очищенные от шкурки) и мелко порубленный чеснок, тушить 5 минут.', default=unknownAnswer, rightAnswer=rightAnswerRecept))
print  (cl.classify('Торт из омлета нечасто встречается на наших столах, и совсем напрасно. Яичные блинчики можно перемазать любой начинкой, на усмотрение хозяйки, украсить - и любители омлета будут в восторге! Подайте омлет в виде закусочного торта - и такая холодная закуска станет выигрышным вариантом и для праздничного меню, и для домашнего ужина.', default=unknownAnswer, rightAnswer=rightAnswerRecept))
