#Выведите все позиций списка товаров принадлежащие какой-либо категории с названиями товаров и названиями категорий. Список должен быть отсортирован по названию товара, названию категории. Для соединения таблиц необходимо использовать оператор INNER JOIN.
#Ожидаемый формат результата:
#NB! 
#    Выборки, полученные с помощью оператора SELECT могут быть отсортированы по нескольким атрибутам. Для этого необходимо в операторе ORDER BY указать набор атрибутов через запятую в необходимом порядке.
 #   В запросе для соединения нескольких источников данных операцию соединения можно использовать многократно. Например, для соединения таблиц A, B и C можно использовать запрос вида:
#SELECT * FROM A
#  INNER JOIN B
#    ON A.b_id = B.id
#  INNER JOIN C
#    ON a.c_id = C.id


select good.name as good_name, category.name as category_name
from sale
inner join sale_has_good on sale.id=sale_has_good.sale_id
inner join good on sale_has_good.good_id=good.id
inner join category_has_good on category_has_good.good_id=good.id
inner join category on category_has_good.category_id=category.id
group by good.name, category.name
order by good.name, category.name
